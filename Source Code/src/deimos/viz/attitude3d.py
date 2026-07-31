"""
attitude3d.py
=============

3D animated attitude visualizer -- the satellite body (schematic cube or the
actual CAD mesh via viz/stl_mesh.py) rotating through a simulated maneuver,
with a telemetry HUD and a dashed ghost at the target attitude.

Ported from the pre-package tree (legacy/AttitudeVisualizer.py) during the
2026-07 restructure and integrated with the deimos pipeline: the
`animate_results()` entry point consumes a SimResults straight from
sim.runner.simulate(), so anything the CLI or a notebook can simulate, it
can animate -- `deimos animate --scenario ... --controller ... --save out.mp4`
produces the demo-video asset directly.

Design notes kept from the original:
  * matplotlib 3D with orthographic projection -- no extra dependency, good
    enough for a body + triad at interactive rates once the frame count is
    strided down (see `stride` in animate_results).
  * The body mesh is swappable: cube by default, any (vertices, faces) pair
    otherwise. The ghost target uses the same mesh as a dashed wireframe.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers projection='3d'
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from deimos.math.quaternion import Quaternion


CUBE_VERTICES: np.ndarray = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1],
], dtype=np.float64) * np.array([0.45, 0.45, 1.0])   # 3U-ish proportions

CUBE_FACES: tuple = (
    (0, 2, 6, 4),
    (1, 3, 7, 5),
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
)

CUBE_FACE_COLORS: tuple = (
    "#3f7cac", "#3f7cac",
    "#c0524f", "#c0524f",
    "#5fa777", "#5fa777",
)

MESH_FACE_COLOR = "#8899aa"          # single steel tone for an STL mesh

AXIS_COLORS = {"x": "#ff4b4b", "y": "#4bff7a", "z": "#4b9bff"}
AXIS_LENGTH: float = 1.6

BG_COLOR = "#0d1117"
PANE_COLOR = (0.08, 0.09, 0.11, 1.0)
GRID_COLOR = "#2b2f36"
TEXT_COLOR = "#e6e6e6"
HUD_FACE_COLOR = "#161b22"
HUD_EDGE_COLOR = "#3a3f47"


class AttitudeVisualizer:
    """Frame-by-frame 3D attitude animation.

    quaternion_history : (T, 4) scalar-first body->inertial quaternions
    omega_history      : optional (T, 3) body rates for the HUD
    t_history          : optional (T,) times for the HUD
    target_quaternion  : optional (4,) -- draws the dashed ghost + error line
    Omega_history      : optional (T, N) wheel speeds for the HUD
    max_wheel_speed    : [rad/s] flags SAT next to a pinned wheel in the HUD
    body               : optional (vertices, faces) mesh replacing the cube
                         (viz/stl_mesh.load_stl_mesh returns this shape)
    """

    def __init__(
        self,
        quaternion_history: Sequence[np.ndarray],
        omega_history: Optional[Sequence[np.ndarray]] = None,
        t_history: Optional[Sequence[float]] = None,
        target_quaternion: Optional[Sequence[float]] = None,
        Omega_history: Optional[Sequence[np.ndarray]] = None,
        max_wheel_speed: Optional[float] = None,
        body: Optional[tuple] = None,
    ) -> None:
        self.q_history = quaternion_history
        self.omega_history = omega_history
        self.t_history = t_history
        self.Omega_history = Omega_history
        self.max_wheel_speed = max_wheel_speed
        self.target_quaternion = (
            np.array(target_quaternion, dtype=np.float64)
            if target_quaternion is not None else None
        )

        if body is not None:
            self.body_vertices, self.body_faces = body
            self.body_facecolors = MESH_FACE_COLOR
        else:
            self.body_vertices, self.body_faces = CUBE_VERTICES, CUBE_FACES
            self.body_facecolors = CUBE_FACE_COLORS

        self._build_figure()
        self._build_body()
        self._build_body_axes()
        self._build_inertial_triad()
        if self.target_quaternion is not None:
            self._build_ghost()
        self._build_hud()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.patch.set_facecolor(BG_COLOR)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_proj_type("ortho")
        self.ax.set_facecolor(BG_COLOR)

        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.set_facecolor(PANE_COLOR)
            pane.set_edgecolor(GRID_COLOR)

        self.ax.xaxis._axinfo["grid"]["color"] = GRID_COLOR
        self.ax.yaxis._axinfo["grid"]["color"] = GRID_COLOR
        self.ax.zaxis._axinfo["grid"]["color"] = GRID_COLOR

        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_box_aspect([1, 1, 1])

        self.ax.set_xlabel("X", color=TEXT_COLOR)
        self.ax.set_ylabel("Y", color=TEXT_COLOR)
        self.ax.set_zlabel("Z", color=TEXT_COLOR)
        self.ax.tick_params(colors=TEXT_COLOR)

        self.title_artist = self.ax.set_title(
            "DEIMoS Attitude", color=TEXT_COLOR, fontsize=12, pad=12
        )

    def _face_polys(self, vertices: np.ndarray) -> list:
        return [[vertices[i] for i in face] for face in self.body_faces]

    def _build_body(self) -> None:
        self.poly_collection = Poly3DCollection(
            self._face_polys(self.body_vertices),
            facecolors=self.body_facecolors,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
        )
        self.ax.add_collection3d(self.poly_collection)

    def _build_ghost(self) -> None:
        R_target = Quaternion(*self.target_quaternion).to_rotation_matrix()
        rotated = self.body_vertices @ R_target.T
        self.ghost_collection = Poly3DCollection(
            self._face_polys(rotated),
            facecolors=(0.0, 0.0, 0.0, 0.0),
            edgecolors=(1.0, 0.82, 0.40, 0.9),
            linewidths=1.2,
            linestyles="--",
        )
        self.ax.add_collection3d(self.ghost_collection)

    def _build_body_axes(self) -> None:
        origin = [0, 0, 0]
        self.quiver_x = self.ax.quiver(*origin, 1, 0, 0, color=AXIS_COLORS["x"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)
        self.quiver_y = self.ax.quiver(*origin, 0, 1, 0, color=AXIS_COLORS["y"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)
        self.quiver_z = self.ax.quiver(*origin, 0, 0, 1, color=AXIS_COLORS["z"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)

    def _build_inertial_triad(self) -> None:
        L = AXIS_LENGTH * 0.9
        for direction in (
            ([0, L], [0, 0], [0, 0]),
            ([0, 0], [0, L], [0, 0]),
            ([0, 0], [0, 0], [0, L]),
        ):
            self.ax.plot(*direction, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6)

    def _build_hud(self) -> None:
        self.telemetry_artist = self.ax.text2D(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            fontsize=9,
            fontfamily="monospace",
            color=TEXT_COLOR,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor=HUD_FACE_COLOR, alpha=0.9, edgecolor=HUD_EDGE_COLOR),
        )

    @staticmethod
    def compute_rotation_matrix(q: np.ndarray) -> np.ndarray:
        return Quaternion(*q).to_rotation_matrix()

    @staticmethod
    def quaternion_error_angle_deg(q: np.ndarray, q_target: np.ndarray) -> float:
        q_target_inv = Quaternion(*q_target).inverse()
        q_current = Quaternion(*q)
        q_err = q_target_inv * q_current
        q_err.normalize()
        w_err = np.clip(abs(q_err.q[0]), -1.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(w_err)))

    def _update_body(self, R: np.ndarray) -> None:
        rotated = self.body_vertices @ R.T
        self.poly_collection.set_verts(self._face_polys(rotated))

    def _update_body_axes(self, R: np.ndarray) -> None:
        self.quiver_x.remove()
        self.quiver_y.remove()
        self.quiver_z.remove()

        origin = [0, 0, 0]
        x_dir = R @ np.array([1, 0, 0])
        y_dir = R @ np.array([0, 1, 0])
        z_dir = R @ np.array([0, 0, 1])

        self.quiver_x = self.ax.quiver(*origin, *x_dir, color=AXIS_COLORS["x"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)
        self.quiver_y = self.ax.quiver(*origin, *y_dir, color=AXIS_COLORS["y"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)
        self.quiver_z = self.ax.quiver(*origin, *z_dir, color=AXIS_COLORS["z"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)

    def _format_telemetry(self, frame: int, q: np.ndarray) -> str:
        w, x, y, z = q
        q_norm = float(np.linalg.norm(q))

        t_str = f"{self.t_history[frame]:7.3f} s" if self.t_history is not None else f"frame {frame}"

        lines = [
            f"t        = {t_str}",
            f"q (wxyz) = [{w:+.4f}, {x:+.4f}, {y:+.4f}, {z:+.4f}]",
            f"|q|      = {q_norm:.6f}",
        ]

        if self.omega_history is not None:
            wx, wy, wz = self.omega_history[frame]
            omega_norm = float(np.linalg.norm(self.omega_history[frame]))
            lines.append(f"omega    = [{wx:+.4f}, {wy:+.4f}, {wz:+.4f}] rad/s")
            lines.append(f"|omega|  = {omega_norm:.4f} rad/s")

        if self.Omega_history is not None:
            Om = np.asarray(self.Omega_history[frame], dtype=float)
            rpm = Om * 60.0 / (2.0 * np.pi)
            sat = ""
            if self.max_wheel_speed is not None and np.any(np.abs(Om) >= self.max_wheel_speed):
                sat = "  SAT!"
            lines.append("wheels   = [" + ", ".join(f"{v:+6.0f}" for v in rpm) + f"] rpm{sat}")

        if self.target_quaternion is not None:
            err_deg = self.quaternion_error_angle_deg(q, self.target_quaternion)
            lines.append(f"att. err = {err_deg:7.3f} deg")

        return "\n".join(lines)

    def update(self, frame: int):
        q = self.q_history[frame]
        R = self.compute_rotation_matrix(q)

        self._update_body(R)
        self._update_body_axes(R)
        self.telemetry_artist.set_text(self._format_telemetry(frame, q))

        artists = [self.poly_collection, self.quiver_x, self.quiver_y,
                   self.quiver_z, self.telemetry_artist]
        if self.target_quaternion is not None:
            artists.append(self.ghost_collection)
        return artists

    def animate(self, interval_ms: int = 20, save_path: Optional[str] = None, fps: int = 30):
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.q_history),
            interval=interval_ms,
            repeat=True,
            blit=False,
        )
        if save_path:
            save_path = str(save_path)
            try:
                anim.save(save_path, writer="ffmpeg", fps=fps,
                          savefig_kwargs={"facecolor": BG_COLOR})
            except (FileNotFoundError, RuntimeError, ValueError):
                # no ffmpeg on PATH -- fall back to an animated GIF via
                # Pillow (a matplotlib dependency), so the deliverable still
                # renders on a machine with nothing extra installed
                gif_path = str(save_path).rsplit(".", 1)[0] + ".gif"
                print(f"[attitude3d] ffmpeg unavailable -- writing {gif_path} instead")
                anim.save(gif_path, writer=PillowWriter(fps=fps),
                          savefig_kwargs={"facecolor": BG_COLOR})
            plt.close(self.fig)
        else:
            plt.show()
        return anim


def animate_results(results, save_path=None, stl_path=None, stride=None,
                     fps: int = 30, max_frames: int = 1200):
    """Animate a SimResults from sim.runner.simulate().

    stride: keep every stride-th sample. Default: whatever makes playback
        real-time at `fps` (stride = 1/(fps*dt)), further increased if needed
        to keep the total under max_frames -- a 3000 s desaturation run
        becomes a time-lapse rather than a 3000 s video.
    stl_path: STL file of the CAD model (viz/stl_mesh.py); None = cube.
    save_path: .mp4 via ffmpeg (GIF fallback when ffmpeg is missing);
        None = interactive window.
    """
    t = np.asarray(results.t)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    if stride is None:
        stride = max(1, round(1.0 / (fps * dt)))
        if len(t) / stride > max_frames:
            stride = int(np.ceil(len(t) / max_frames))

    body = None
    if stl_path is not None:
        from deimos.viz.stl_mesh import load_stl_mesh
        body = load_stl_mesh(stl_path)

    viz = AttitudeVisualizer(
        quaternion_history=np.asarray(results.q)[::stride],
        omega_history=np.asarray(results.omega)[::stride],
        t_history=t[::stride],
        target_quaternion=results.q_target,
        Omega_history=np.asarray(results.Omega)[::stride],
        max_wheel_speed=results.max_speed,
        body=body,
    )
    viz.title_artist.set_text(f"DEIMoS Attitude — {results.name}")
    return viz.animate(interval_ms=int(1000 / fps), save_path=save_path, fps=fps)
