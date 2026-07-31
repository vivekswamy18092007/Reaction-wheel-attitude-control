"""
stability.py
=============

Whether a (K, D, mu) triple a controller was actually built with satisfies
Wie/Weiss/Arapostathis's Sec. III global-stability proof, plus the
linearized closed-loop modes used to sanity-check gain choices.

Linearization used for the closed-loop modes
--------------------------------------------
Near the target, q_ev ~ (1/2) * theta, so both PD and Wie's law reduce to

    J theta_ddot + D theta_dot + (K/2) theta = 0

giving, per axis i,

    omega_n,i = sqrt(K_ii / (2 J_ii))        zeta_i = D_ii / (2 sqrt(J_ii K_ii / 2))

Note the factor of 1/2 -- it comes from the quaternion vector part being
half-angle, and dropping it inflates omega_n by sqrt(2). Substituting Wie's
design relations (K = k*J, D = d*J, k = 2*omega_n^2, d = 2*zeta*omega_n)
recovers omega_n,i = omega_n and zeta_i = zeta exactly, which is the check
that this linearization matches the paper's intent.
"""

from __future__ import annotations

import numpy as np


def closed_loop_modes(K: np.ndarray, D: np.ndarray, J: np.ndarray):
    """Per-axis (omega_n [rad/s], zeta [-]) of the linearized closed loop."""
    Kd_, Dd_, Jd_ = np.diag(K), np.diag(D), np.diag(J)
    omega_n = np.sqrt(Kd_ / (2.0 * Jd_))
    zeta = Dd_ / (2.0 * np.sqrt(Jd_ * Kd_ / 2.0))
    return omega_n, zeta


def stability_branch(K: np.ndarray, D: np.ndarray, J: np.ndarray, mu: float):
    """
    Which branch of Wie/Weiss/Arapostathis Sec. III global-stability proof this
    (K, D, mu) triple satisfies, and the numerical evidence for it.

    The proof needs EITHER mu = 1 with any K = K^T > 0, D = D^T > 0,
    OR mu = 0 with K^-1 = alpha*J + beta*I for some alpha, beta >= 0.
    A mu = 0 controller with an unmatched K has no guarantee from this proof.
    """
    sym_K = np.allclose(K, K.T, atol=1e-12)
    sym_D = np.allclose(D, D.T, atol=1e-12)
    pos_K = np.all(np.linalg.eigvalsh(0.5 * (K + K.T)) > 0)
    pos_D = np.all(np.linalg.eigvalsh(0.5 * (D + D.T)) > 0)
    basic = sym_K and sym_D and pos_K and pos_D

    if np.isclose(mu, 1.0):
        ok = basic
        detail = ("mu = 1: gyroscopic term cancelled in closed loop; any "
                  "K = K^T > 0, D = D^T > 0 suffices.")
        return ("GUARANTEED" if ok else "NOT SATISFIED"), detail, None

    if np.isclose(mu, 0.0):
        # Least-squares fit of K^-1 against the span of {J, I}; the residual
        # says whether K^-1 has the required *form*, and the signs of
        # (alpha, beta) say whether it lands in the admissible cone.
        Kinv = np.linalg.inv(K)
        A = np.column_stack([J.ravel(), np.eye(3).ravel()])
        coef, *_ = np.linalg.lstsq(A, Kinv.ravel(), rcond=None)
        alpha, beta = coef
        residual = np.linalg.norm(Kinv - (alpha * J + beta * np.eye(3)))
        rel = residual / max(np.linalg.norm(Kinv), 1e-300)
        in_span = rel < 1e-6
        nonneg = alpha >= -1e-12 and beta >= -1e-12

        detail = (f"mu = 0: requires K^-1 = alpha*J + beta*I with alpha, beta >= 0.\n"
                  f"        fitted alpha = {alpha:+.6e}\n"
                  f"        fitted beta  = {beta:+.6e}\n"
                  f"        relative residual |K^-1 - (alpha*J + beta*I)| = {rel:.2e}")

        if not basic:
            status = "NOT SATISFIED"
        elif in_span and nonneg:
            status = "GUARANTEED"
        elif in_span and not nonneg:
            # K^-1 has exactly the right form but sits outside the admissible
            # cone. Worth separating from the "wrong form entirely" case: the
            # controller is still a valid symmetric-positive-definite design
            # and may well converge, but this proof does not cover it.
            status = "NOT ESTABLISHED (alpha < 0)"
            detail += ("\n        K^-1 is exactly in span{J, I} (residual at machine\n"
                       "        precision) but alpha < 0, so it falls outside the\n"
                       "        alpha, beta >= 0 cone the Sec. III proof requires.\n"
                       "        Note this is structural, not a tuning artifact: the\n"
                       "        least-squares fit in design() matches 1/J_i against\n"
                       "        a*J_i + b, and 1/J_i is decreasing in J_i, so the\n"
                       "        slope is negative for ANY J with distinct moments --\n"
                       "        including the paper's own Sec. VI example.\n"
                       "        K = K^T > 0 still holds, so this is 'unproven by this\n"
                       "        route', NOT 'shown unstable'. Verify convergence\n"
                       "        empirically and say so explicitly in the report.")
        else:
            status = "NOT GUARANTEED (K unmatched to J)"
        return status, detail, (alpha, beta)

    return ("UNKNOWN",
            f"mu = {mu} is neither 0 nor 1; the Sec. III proof covers only "
            f"those two cases.", None)


def offdiag_materiality(J: np.ndarray, threshold: float = 0.01):
    """
    How much WieRegulator.design()'s diag(J)-only gain fitting is missing by
    ignoring J's off-diagonal (products-of-inertia) terms.

    This matters for exactly two of the four design() cases:
      - "mortensen"      K = k*diag(1/J_i)       -- off-diagonals dropped entirely
      - "near_eigenaxis" alpha,beta fitted against Ji = diag(J) only, i.e.
                         Eq. 25's least-squares problem is posed per principal
                         axis (3 equations), not as a full 3x3 matrix fit.
                         (The resulting K = inv(alpha*J + beta*I) DOES use the
                         full J when forming K itself -- only the alpha/beta
                         FIT is diagonal-only.)
    "eigenaxis" (K=k*J) already uses the full J directly, and "robust" (K=k*I)
    doesn't depend on J's shape at all, so this check does not apply to them.

    Returns (ratio, material, note):
      ratio    = ||J - diag(diag(J))||_F / ||diag(diag(J))||_F
      material = ratio > threshold (1% by default, matching the
                 "diagonal-dominant" check already printed in the PLANT block)
      note     = human-readable verdict with the number backing it up
    """
    off = J - np.diag(np.diag(J))
    ratio = np.linalg.norm(off) / np.linalg.norm(np.diag(np.diag(J)))
    material = ratio > threshold
    if material:
        note = (f"off-diagonal/diag ratio = {ratio:.2e} EXCEEDS the {threshold:.0%} "
                f"threshold => design()'s diag(J)-only fit (mortensen, "
                f"near_eigenaxis) is a KNOWN LIMITATION for this J.")
    else:
        note = (f"off-diagonal/diag ratio = {ratio:.2e} is under the {threshold:.0%} "
                f"threshold => design()'s diag(J)-only fit (mortensen, "
                f"near_eigenaxis) is justified for this J.")
    return ratio, material, note
