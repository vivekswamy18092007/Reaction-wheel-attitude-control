# Polaris attitude visualizer - cube + body axes + ghost target cube + HUD
# faces update in place (set_verts) instead of clearing axes every frame,
# this was the biggest perf win - ax.clear() rebuilds everything from scratch

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # needed for projection='3d' to work
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from QuaternionLibrary import Quaternion


# cube geometry - doesn't change, no point recomputing per instance
CUBE_VERTICES: np.ndarray = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1],
], dtype=np.float64)

# 4 vertex indices per face, order matters or the quad renders twisted
CUBE_FACES: tuple = (
    (0, 2, 6, 4),  # bottom z=-1
    (1, 3, 7, 5),  # top z=+1
    (0, 1, 3, 2),  # left x=-1
    (4, 5, 7, 6),  # right x=+1
    (0, 1, 5, 4),  # front y=-1
    (2, 3, 7, 6),  # back y=+1
)

# opposite faces same color so each axis reads as one color from any angle
FACE_COLORS: tuple = (
    "#3f7cac", "#3f7cac",   # X faces, blue
    "#c0524f", "#c0524f",   # Y faces, red
    "#5fa777", "#5fa777",   # Z faces, green
)

AXIS_COLORS = {"x": "#ff4b4b", "y": "#4bff7a", "z": "#4b9bff"}
AXIS_LENGTH: float = 1.6

# dark theme colors, basically just picked stuff that looked decent
BG_COLOR = "#0d1117"
PANE_COLOR = (0.08, 0.09, 0.11, 1.0)
GRID_COLOR = "#2b2f36"
TEXT_COLOR = "#e6e6e6"
HUD_FACE_COLOR = "#161b22"
HUD_EDGE_COLOR = "#3a3f47"


class AttitudeVisualizer:
    # quaternion_history: list of [w,x,y,z] per frame, from AttitudeSimulator.history["q"]
    # omega_history / t_history: optional, just for the HUD readout
    # target_quaternion: if given, draws a ghost cube + shows attitude error in HUD

    def __init__(
        self,
        quaternion_history: Sequence[np.ndarray],
        omega_history: Optional[Sequence[np.ndarray]] = None,
        t_history: Optional[Sequence[float]] = None,
        target_quaternion: Optional[Sequence[float]] = None,
    ) -> None:
        self.q_history = quaternion_history
        self.omega_history = omega_history
        self.t_history = t_history
        self.target_quaternion = (
            np.array(target_quaternion, dtype=np.float64)
            if target_quaternion is not None else None
        )

        self._build_figure()
        self._build_cube()
        self._build_body_axes()
        self._build_inertial_triad()
        if self.target_quaternion is not None:
            self._build_ghost_cube()
        self._build_hud()

    # ---- one time setup, runs once in __init__ ----

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.patch.set_facecolor(BG_COLOR)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_proj_type("ortho")  # was looking like a parallelepiped without this, perspective distortion
        self.ax.set_facecolor(BG_COLOR)

        # have to dig into _axinfo to recolor the grid, no clean public api for it afaik
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.set_facecolor(PANE_COLOR)
            pane.set_edgecolor(GRID_COLOR)

        self.ax.xaxis._axinfo["grid"]["color"] = GRID_COLOR
        self.ax.yaxis._axinfo["grid"]["color"] = GRID_COLOR
        self.ax.zaxis._axinfo["grid"]["color"] = GRID_COLOR

        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_box_aspect([1, 1, 1])  # without this the cube stretches even with equal limits

        self.ax.set_xlabel("X", color=TEXT_COLOR)
        self.ax.set_ylabel("Y", color=TEXT_COLOR)
        self.ax.set_zlabel("Z", color=TEXT_COLOR)
        self.ax.tick_params(colors=TEXT_COLOR)

        self.title_artist = self.ax.set_title(
            "Attitude Visualizer", color=TEXT_COLOR, fontsize=12, pad=12
        )

    def _build_cube(self) -> None:
        # one Poly3DCollection, created once. update() just mutates its verts,
        # never recreate this - that was the whole point of the perf fix
        initial_faces = [[CUBE_VERTICES[i] for i in face] for face in CUBE_FACES]
        self.poly_collection = Poly3DCollection(
            initial_faces,
            facecolors=FACE_COLORS,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.8,
        )
        self.ax.add_collection3d(self.poly_collection)

    def _build_ghost_cube(self) -> None:
        # target attitude, fixed, drawn once and never touched again
        # NOTE: don't pass alpha= to the collection here, it overrides the
        # per-color alpha and you get a solid black box instead of an outline
        # (wasted an hour on this, alpha has to be baked into the RGBA tuple)
        R_target = Quaternion(*self.target_quaternion).to_rotation_matrix()
        rotated = CUBE_VERTICES @ R_target.T
        ghost_faces = [[rotated[i] for i in face] for face in CUBE_FACES]

        self.ghost_collection = Poly3DCollection(
            ghost_faces,
            facecolors=(0.0, 0.0, 0.0, 0.0),
            edgecolors=(1.0, 0.82, 0.40, 0.9),  # yellow, alpha baked in
            linewidths=1.4,
            linestyles="--",
        )
        self.ax.add_collection3d(self.ghost_collection)

    def _build_body_axes(self) -> None:
        # quiver doesn't have set_verts or anything like it, so these get
        # removed + redrawn every frame in _update_body_axes. not ideal but
        # it's only 3 small arrows so it's not really a bottleneck
        origin = [0, 0, 0]
        self.quiver_x = self.ax.quiver(*origin, 1, 0, 0, color=AXIS_COLORS["x"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)
        self.quiver_y = self.ax.quiver(*origin, 0, 1, 0, color=AXIS_COLORS["y"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)
        self.quiver_z = self.ax.quiver(*origin, 0, 0, 1, color=AXIS_COLORS["z"],
                                        length=AXIS_LENGTH, normalize=True, linewidth=2)

    def _build_inertial_triad(self) -> None:
        # fixed world frame axes so there's something on screen that ISN'T
        # spinning, otherwise hard to tell how fast the cube is actually rotating
        L = AXIS_LENGTH * 0.9
        for direction in (
            ([0, L], [0, 0], [0, 0]),
            ([0, 0], [0, L], [0, 0]),
            ([0, 0], [0, 0], [0, L]),
        ):
            self.ax.plot(*direction, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6)

    def _build_hud(self) -> None:
        # text2D + transAxes pins this to the corner of the screen, regardless
        # of cube rotation. using ax.text() instead puts it IN the 3D scene
        # and it rotates/shrinks with the view which is not what we want
        self.telemetry_artist = self.ax.text2D(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            fontsize=9,
            fontfamily="monospace",  # keeps the number columns lined up
            color=TEXT_COLOR,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor=HUD_FACE_COLOR, alpha=0.9, edgecolor=HUD_EDGE_COLOR),
        )

    # stuff that runs every frame

    @staticmethod
    def compute_rotation_matrix(q: np.ndarray) -> np.ndarray:
        return Quaternion(*q).to_rotation_matrix()

    @staticmethod
    def quaternion_error_angle_deg(q: np.ndarray, q_target: np.ndarray) -> float:
        # q_e = q_target^-1 (x) q, then angle = 2*arccos(|q_e0|)
        # abs() because q and -q are literally the same orientation,
        # don't want the error to jump around because of sign flips
        q_target_inv = Quaternion(*q_target).inverse()
        q_current = Quaternion(*q)
        q_err = q_target_inv * q_current
        q_err.normalize()
        w_err = np.clip(abs(q_err.q[0]), -1.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(w_err)))

    def _update_cube(self, R: np.ndarray) -> None:
        rotated_vertices = CUBE_VERTICES @ R.T
        face_polys = [[rotated_vertices[i] for i in face] for face in CUBE_FACES]
        self.poly_collection.set_verts(face_polys)  # in place, no rebuild

    def _update_body_axes(self, R: np.ndarray) -> None:
        self.quiver_x.remove()
        self.quiver_y.remove()
        self.quiver_z.remove()

        origin = [0, 0, 0]
        # R @ unit vector = where that body axis is currently pointing in
        # the inertial frame, same logic as rotating the cube verts
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
        q_norm = float(np.linalg.norm(q))  # should always be ~1.0, good sanity check

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

        if self.target_quaternion is not None:
            err_deg = self.quaternion_error_angle_deg(q, self.target_quaternion)
            lines.append(f"att. err = {err_deg:7.3f} deg")  # this is what the controller should drive to 0

        return "\n".join(lines)

    def update(self, frame: int):
        q = self.q_history[frame]
        R = self.compute_rotation_matrix(q)

        self._update_cube(R)
        self._update_body_axes(R)
        self.telemetry_artist.set_text(self._format_telemetry(frame, q))

        # returning these so blit=True works later if I bother switching it on
        artists = [self.poly_collection, self.quiver_x, self.quiver_y,
                   self.quiver_z, self.telemetry_artist]
        if self.target_quaternion is not None:
            artists.append(self.ghost_collection)
        return artists

    #  entry point

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
            # need facecolor here too or the saved mp4 background comes out white
            # even though the figure itself is dark
            anim.save(save_path, writer="ffmpeg", fps=fps,
                      savefig_kwargs={"facecolor": BG_COLOR})
            plt.close(self.fig)
        else:
            plt.show()
        return anim
