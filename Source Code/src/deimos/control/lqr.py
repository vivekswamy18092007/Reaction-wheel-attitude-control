"""
LQR controller -- not implemented yet.

Placeholder so registry.py has somewhere to point "lqr" at without an
if/elif edit once this is written for real: implement a class here matching
control.base.Controller (compute_torque(q, omega, q_target, dt=None),
reset(), Ki, u_max attributes) and swap build_lqr's body below.
"""


class LQRController:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LQR controller is not implemented yet. Implement LQRController "
            "here (matching control.base.Controller) and update "
            "control/registry.py's build_lqr()."
        )
