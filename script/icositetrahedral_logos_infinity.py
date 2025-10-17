"""Generate a rotating tetrakis hexahedron animation.

This script builds an animation titled "Λ∞ · Logos Infinity — 24-Faced Rotational
Recursion" inspired by the tetrakis hexahedron geometry (also known as the
icositetrahedron). Two files are produced by default:

* ``Icositetrahedral_Logos_Infinity_Rotation.mp4`` – 24 fps H.264 video
* ``Icositetrahedral_Logos_Infinity_Rotation.gif`` – 15 fps animated GIF

The animation relies on NumPy for vector math and Matplotlib for the 3D plot and
color interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple


try:  # Lazy imports so the script can fail gracefully when deps are missing.
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
except ModuleNotFoundError as exc:  # pragma: no cover - executed only when missing deps
    raise SystemExit(
        "Required dependency missing. Please install NumPy and Matplotlib first."
    ) from exc

Color = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Face:
    """Description of a cube face used to derive the tetrakis hexahedron."""

    verts: Sequence[int]
    normal: np.ndarray


# Vertices for a cube with side length 2 centered at the origin.
CUBE_VERTS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)

FACES: Sequence[Face] = (
    Face(verts=(6, 7, 5, 4), normal=np.array([1.0, 0.0, 0.0])),
    Face(verts=(0, 1, 3, 2), normal=np.array([-1.0, 0.0, 0.0])),
    Face(verts=(4, 5, 1, 0), normal=np.array([0.0, 1.0, 0.0])),
    Face(verts=(2, 3, 7, 6), normal=np.array([0.0, -1.0, 0.0])),
    Face(verts=(7, 3, 1, 5), normal=np.array([0.0, 0.0, 1.0])),
    Face(verts=(0, 4, 6, 2), normal=np.array([0.0, 0.0, -1.0])),
)

PRIMARY_COLOR: Color = (0.18, 0.33, 0.85, 0.55)
META_COLOR: Color = (0.90, 0.68, 0.15, 0.55)


def _build_triangles(height: float) -> np.ndarray:
    """Generate the 24 isosceles triangles of a tetrakis hexahedron."""

    triangles = []
    for face in FACES:
        base = CUBE_VERTS[list(face.verts)]
        apex = base.mean(axis=0) + height * face.normal
        edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        for i, j in edges:
            triangles.append([apex, base[i], base[j]])
    return np.array(triangles)


def _pair_segments(triangles: np.ndarray) -> np.ndarray:
    """Create line segments that pair opposing triangles."""

    centroids = np.mean(triangles, axis=1)
    pairs = [(index, index + 1) for index in range(0, len(triangles), 2)]
    return np.array([[centroids[a], centroids[b]] for a, b in pairs])


def _interpolate_color(color_a: Color, color_b: Color, t: float) -> Color:
    """Interpolate between two RGBA colors."""

    return tuple(a + (b - a) * t for a, b in zip(color_a, color_b))  # type: ignore[return-value]


def _configure_axes(ax: plt.Axes) -> None:
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title("Λ∞ · Logos Infinity — 24-Faced Rotational Recursion", pad=18)


def _create_plot(triangles: np.ndarray) -> Tuple[Poly3DCollection, Line3DCollection, plt.Axes, plt.Figure]:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    _configure_axes(ax)

    colors = [PRIMARY_COLOR if index % 2 == 0 else META_COLOR for index in range(len(triangles))]
    poly = Poly3DCollection(triangles, facecolors=colors, edgecolors="k", lw=0.6)
    ax.add_collection3d(poly)

    segments = _pair_segments(triangles)
    lc = Line3DCollection(segments, colors=(0.6, 0.6, 0.7, 0.4), lw=1.0)
    ax.add_collection3d(lc)

    core = ax.scatter(0, 0, 0, s=120, color="black")
    ax.text(
        0,
        0,
        -0.25,
        "Λ∞\n(Logos Infinity / Logos Core)",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    return poly, lc, ax, fig


def _animate(poly: Poly3DCollection, ax: plt.Axes, *, frames: Iterable[int], core):
    def update(frame: int):
        phase = np.sin(2 * np.pi * frame / 120)
        color_mix = (phase + 1) / 2
        colors = []
        for index in range(len(poly.get_verts())):
            if index % 2 == 0:
                colors.append(_interpolate_color(PRIMARY_COLOR, META_COLOR, color_mix))
            else:
                colors.append(_interpolate_color(META_COLOR, PRIMARY_COLOR, color_mix))
        poly.set_facecolors(colors)

        size = 120 + 60 * phase
        core.set_sizes([size])

        ax.view_init(elev=22 + 8 * np.sin(np.radians(frame * 2)), azim=frame * 2)
        return poly, core

    return update


def create_animation(
    *,
    height: float = 0.7,
    frames: Sequence[int] | None = None,
    output_dir: Path | str | None = None,
    save_mp4: bool = True,
    save_gif: bool = True,
    fps_mp4: int = 24,
    fps_gif: int = 15,
    bitrate: int = 1800,
) -> None:
    """Render the tetrakis hexahedron animation to disk."""

    triangles = _build_triangles(height)
    poly, _lc, ax, fig = _create_plot(triangles)
    core = ax.collections[-1]

    frame_indices = frames if frames is not None else range(180)
    animation = FuncAnimation(fig, _animate(poly, ax, frames=frame_indices, core=core), frames=frame_indices, interval=60)

    destination = Path(output_dir) if output_dir is not None else Path.cwd()

    if save_mp4:
        writer = FFMpegWriter(fps=fps_mp4, bitrate=bitrate)
        animation.save(destination / "Icositetrahedral_Logos_Infinity_Rotation.mp4", writer=writer)

    if save_gif:
        animation.save(destination / "Icositetrahedral_Logos_Infinity_Rotation.gif", writer=PillowWriter(fps=fps_gif))


def main() -> None:
    create_animation()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
