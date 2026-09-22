from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "paperI_fig1_diffusion_framework_ja"
SHOW_FIGURE = False


# -----------------------------------------------------------------------------
# Font configuration
# -----------------------------------------------------------------------------
def configure_japanese_font() -> str:
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "Yu Gothic",
        "Meiryo",
        "Hiragino Sans",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": selected,
            "font.size": 10.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    return selected


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
LINE_COLOR = "0.15"
SUBTLE_COLOR = "0.40"
BOX_FILL = "0.985"
BRANCH_FILL = "0.955"
TAG_FILL = "0.92"
DASH_FILL = "0.975"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2.0)

    @property
    def right(self) -> tuple[float, float]:
        return (self.x + self.w, self.y + self.h / 2.0)

    @property
    def top(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def rounded_box(
    ax: plt.Axes,
    box: Box,
    *,
    fill: str = BOX_FILL,
    linewidth: float = 1.2,
    linestyle: str = "-",
    rounding: float = 0.012,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=LINE_COLOR,
        facecolor=fill,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3,rad=0.0",
    linewidth: float = 1.25,
    linestyle: str = "-",
    mutation_scale: float = 12.0,
    zorder: int = 3,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=LINE_COLOR,
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text_center(
    ax: plt.Axes,
    box: Box,
    text: str,
    *,
    fontsize: float = 10.5,
    weight: str = "normal",
    y_offset: float = 0.0,
    linespacing: float = 1.25,
    zorder: int = 4,
) -> None:
    ax.text(
        box.x + box.w / 2.0,
        box.y + box.h / 2.0 + y_offset,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=linespacing,
        color=LINE_COLOR,
        zorder=zorder,
    )


def small_tag(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    width: float,
    height: float = 0.047,
) -> Box:
    box = Box(x, y, width, height)
    rounded_box(ax, box, fill=TAG_FILL, linewidth=0.9, rounding=0.010, zorder=3)
    text_center(ax, box, text, fontsize=8.8, weight="medium")
    return box


def add_header(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color=LINE_COLOR,
        zorder=5,
    )


def draw_point_cloud(ax: plt.Axes, box: Box) -> None:
    rng = np.random.default_rng(20260813)
    n1, n2, n3 = 26, 20, 12
    p1 = rng.normal(loc=(0.35, 0.56), scale=(0.18, 0.17), size=(n1, 2))
    p2 = rng.normal(loc=(0.70, 0.40), scale=(0.14, 0.13), size=(n2, 2))
    p3 = rng.normal(loc=(0.63, 0.76), scale=(0.10, 0.08), size=(n3, 2))
    pts = np.vstack([p1, p2, p3])
    pts = np.clip(pts, 0.06, 0.94)

    px = box.x + 0.07 * box.w + pts[:, 0] * 0.86 * box.w
    py = box.y + 0.09 * box.h + pts[:, 1] * 0.72 * box.h
    ax.plot(
        px,
        py,
        linestyle="none",
        marker="o",
        markersize=3.1,
        markerfacecolor="white",
        markeredgecolor=LINE_COLOR,
        markeredgewidth=0.65,
        zorder=4,
    )

    qx = box.x + np.array([0.20, 0.52, 0.82]) * box.w
    qy = box.y + np.array([0.23, 0.68, 0.30]) * box.h
    ax.plot(
        qx,
        qy,
        linestyle="none",
        marker="x",
        markersize=4.8,
        markeredgewidth=1.0,
        color=LINE_COLOR,
        zorder=4,
    )

    ax.text(
        box.x + box.w / 2.0,
        box.y + box.h - 0.027,
        "非一様な銀河点群",
        ha="center",
        va="top",
        fontsize=9.1,
        fontweight="medium",
        color=LINE_COLOR,
        zorder=5,
    )
    ax.text(
        box.x + box.w / 2.0,
        box.y + 0.025,
        r"観測点 $\mathbf{x}_i$ ／ query 点 $\mathbf{x}_q$",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=SUBTLE_COLOR,
        zorder=5,
    )


def draw_kernel_neighborhood(ax: plt.Axes, box: Box) -> None:
    rng = np.random.default_rng(314159)
    center = np.array([box.x + 0.50 * box.w, box.y + 0.47 * box.h])
    angles = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    radii = np.array([0.19, 0.31, 0.23, 0.37, 0.27, 0.18, 0.34, 0.24, 0.39, 0.22, 0.30, 0.17, 0.33])
    jitter = rng.normal(scale=0.025, size=(angles.size, 2))
    pts = np.column_stack(
        [
            center[0] + box.w * radii * np.cos(angles),
            center[1] + box.h * radii * np.sin(angles),
        ]
    ) + jitter * np.array([box.w, box.h])

    for index, point in enumerate(pts):
        width = 0.55 + 1.1 * (1.0 - radii[index] / radii.max())
        ax.plot(
            [center[0], point[0]],
            [center[1], point[1]],
            linewidth=width,
            color="0.45",
            alpha=0.65,
            zorder=3,
        )

    ax.plot(
        pts[:, 0],
        pts[:, 1],
        linestyle="none",
        marker="o",
        markersize=3.3,
        markerfacecolor="white",
        markeredgecolor=LINE_COLOR,
        markeredgewidth=0.65,
        zorder=4,
    )
    ax.plot(
        center[0],
        center[1],
        linestyle="none",
        marker="o",
        markersize=6.0,
        markerfacecolor="0.80",
        markeredgecolor=LINE_COLOR,
        markeredgewidth=0.9,
        zorder=5,
    )

    ax.text(
        box.x + box.w / 2.0,
        box.y + box.h - 0.027,
        "可変帯域幅カーネル",
        ha="center",
        va="top",
        fontsize=9.1,
        fontweight="medium",
        color=LINE_COLOR,
        zorder=5,
    )
    ax.text(
        box.x + box.w / 2.0,
        box.y + 0.023,
        r"$K_{ij}=\exp[-\|\mathbf{x}_i-\mathbf{x}_j\|^2/(\rho_i\rho_j)]$",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=SUBTLE_COLOR,
        zorder=5,
    )


def build_figure() -> tuple[plt.Figure, plt.Axes]:
    configure_japanese_font()

    fig = plt.figure(figsize=(16.6, 7.6), constrained_layout=False)
    ax = fig.add_axes([0.012, 0.025, 0.976, 0.955])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Main boxes.
    input_box = Box(0.020, 0.325, 0.140, 0.405)
    kernel_box = Box(0.195, 0.325, 0.145, 0.405)
    operator_box = Box(0.375, 0.325, 0.165, 0.405)

    upper_group = Box(0.572, 0.565, 0.410, 0.385)
    lower_group = Box(0.572, 0.185, 0.410, 0.315)

    spectrum_box = Box(0.595, 0.650, 0.125, 0.185)
    fit_box = Box(0.748, 0.650, 0.145, 0.185)
    field_output_box = Box(0.916, 0.632, 0.056, 0.220)

    carre_box = Box(0.595, 0.255, 0.125, 0.165)
    calibration_box = Box(0.748, 0.255, 0.145, 0.165)
    diff_output_box = Box(0.916, 0.238, 0.056, 0.200)

    for box in (input_box, kernel_box, operator_box):
        rounded_box(ax, box, fill=BOX_FILL)

    rounded_box(ax, upper_group, fill=BRANCH_FILL, linewidth=1.1, rounding=0.014, zorder=1)
    rounded_box(ax, lower_group, fill=BRANCH_FILL, linewidth=1.1, rounding=0.014, zorder=1)

    for box in (spectrum_box, fit_box, field_output_box, carre_box, calibration_box, diff_output_box):
        rounded_box(ax, box, fill=BOX_FILL, linewidth=1.0)

    # Input and kernel schematics.
    draw_point_cloud(ax, input_box)
    draw_kernel_neighborhood(ax, kernel_box)

    # Operator box.
    ax.text(
        operator_box.x + operator_box.w / 2.0,
        operator_box.y + operator_box.h - 0.032,
        "Markov作用素と生成作用素",
        ha="center",
        va="top",
        fontsize=10.1,
        fontweight="bold",
        color=LINE_COLOR,
    )
    operator_text = (
        r"$K^{(\alpha)}_{ij}=K_{ij}/(q_i^\alpha q_j^\alpha)$"
        "\n"
        r"$P_{ij}=K^{(\alpha)}_{ij}/d_i^{(\alpha)}$"
        "\n"
        r"$L_\tau=(P-I)/\tau$"
        "\n\n"
        "点群が支持する\n関数空間と局所遷移"
    )
    ax.text(
        operator_box.x + operator_box.w / 2.0,
        operator_box.y + operator_box.h / 2.0 - 0.008,
        operator_text,
        ha="center",
        va="center",
        fontsize=8.9,
        linespacing=1.38,
        color=LINE_COLOR,
    )

    # Branch headers.
    add_header(ax, upper_group.x + 0.018, upper_group.y + upper_group.h - 0.027, "A. 大域的な場表現と正則化")
    add_header(ax, lower_group.x + 0.018, lower_group.y + lower_group.h - 0.027, "B. 局所微分構造と計量較正")

    # Upper branch.
    text_center(
        ax,
        spectrum_box,
        "拡散スペクトル\n"
        r"$\lbrace\phi_a,\eta_a\rbrace$"
        "\n低周波関数基底",
        fontsize=9.3,
        weight="medium",
    )
    text_center(
        ax,
        fit_box,
        "デカルト成分を保持した\n"
        r"$\widehat{\mathbf{v}}(\mathbf{x})$"
        "\n"
        r"$=\sum_a\mathbf{b}_a\phi_a(\mathbf{x})$"
        "\n正則化付き係数推定",
        fontsize=8.55,
        weight="medium",
    )
    text_center(
        ax,
        field_output_box,
        "再構成場\n\n観測点\n＋\nquery 点",
        fontsize=8.4,
        weight="medium",
    )

    # Lower branch.
    text_center(
        ax,
        carre_box,
        "中心化\ncarré du champ\n"
        r"$\widehat\Gamma_i(f,h)$",
        fontsize=9.1,
        weight="medium",
    )
    text_center(
        ax,
        calibration_box,
        "局所 Gram 行列と未較正ベクトル\n"
        r"$\mathbf{G}_i,\mathbf{g}_i(f)$"
        "\n"
        r"$\widehat{\nabla} f=\mathbf{G}_i^\dagger\mathbf{g}_i(f)$",
        fontsize=8.45,
        weight="medium",
    )
    text_center(
        ax,
        diff_output_box,
        "勾配\nJacobian\n勾配部分空間",
        fontsize=8.3,
        weight="medium",
    )

    # Main arrows.
    arrow(ax, input_box.right, kernel_box.left)
    arrow(ax, kernel_box.right, operator_box.left)
    arrow(
        ax,
        operator_box.right,
        spectrum_box.left,
        connectionstyle="arc3,rad=-0.18",
    )
    arrow(
        ax,
        operator_box.right,
        carre_box.left,
        connectionstyle="arc3,rad=0.18",
    )
    arrow(ax, spectrum_box.right, fit_box.left)
    arrow(ax, fit_box.right, field_output_box.left)
    arrow(ax, carre_box.right, calibration_box.left)
    arrow(ax, calibration_box.right, diff_output_box.left)

    # Out-of-sample extension band linking both branches.
    extension_band = Box(0.720, 0.510, 0.250, 0.045)
    rounded_box(
        ax,
        extension_band,
        fill=DASH_FILL,
        linewidth=0.9,
        linestyle="--",
        rounding=0.010,
        zorder=2,
    )
    text_center(
        ax,
        extension_band,
        "Nyström 固有関数拡張と query 点での局所微分拡張",
        fontsize=8.35,
        weight="medium",
    )
    arrow(
        ax,
        (fit_box.x + fit_box.w * 0.72, fit_box.y),
        (extension_band.x + extension_band.w * 0.40, extension_band.y + extension_band.h),
        connectionstyle="arc3,rad=0.08",
        linewidth=0.9,
        linestyle="--",
        mutation_scale=9.0,
    )
    arrow(
        ax,
        (calibration_box.x + calibration_box.w * 0.72, calibration_box.y + calibration_box.h),
        (extension_band.x + extension_band.w * 0.60, extension_band.y),
        connectionstyle="arc3,rad=-0.08",
        linewidth=0.9,
        linestyle="--",
        mutation_scale=9.0,
    )

    # Weighting tags and statistical roles.
    tag_geometry = small_tag(ax, 0.413, 0.250, "幾何重み", width=0.090)
    tag_loss = small_tag(ax, 0.773, 0.585, "損失重み", width=0.091)
    tag_eval = small_tag(ax, 0.875, 0.137, "評価重み", width=0.090, height=0.040)

    arrow(
        ax,
        tag_geometry.top,
        (operator_box.x + operator_box.w * 0.50, operator_box.y),
        linewidth=0.85,
        linestyle=":",
        mutation_scale=8.0,
    )
    arrow(
        ax,
        tag_loss.top,
        (fit_box.x + fit_box.w * 0.50, fit_box.y),
        linewidth=0.85,
        linestyle=":",
        mutation_scale=8.0,
    )
    arrow(
        ax,
        tag_eval.top,
        (0.920, 0.125),
        linewidth=0.85,
        linestyle=":",
        mutation_scale=8.0,
    )

    # Evaluation footer.
    footer = Box(0.570, 0.035, 0.412, 0.090)
    rounded_box(ax, footer, fill=BOX_FILL, linewidth=1.0, rounding=0.012, zorder=2)
    footer_text = (
        "検証設計：学習／検証／観測点上の保持テスト／完全標本 query テスト\n"
        "幾何・損失・評価の重み付けは，互いに異なる推定対象を持つ"
    )
    text_center(ax, footer, footer_text, fontsize=8.9, weight="medium", linespacing=1.35)

    return fig, ax


def save_figure(fig: plt.Figure) -> list[Path]:
    outputs = [
        OUTPUT_DIR / f"{OUTPUT_STEM}.pdf",
        OUTPUT_DIR / f"{OUTPUT_STEM}.png",
        OUTPUT_DIR / f"{OUTPUT_STEM}.svg",
    ]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[1], dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.04)
    return outputs


def main() -> None:
    font_name = configure_japanese_font()
    fig, _ = build_figure()
    outputs = save_figure(fig)
    print(f"Japanese font: {font_name}")
    for path in outputs:
        print(f"Saved: {path}")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
