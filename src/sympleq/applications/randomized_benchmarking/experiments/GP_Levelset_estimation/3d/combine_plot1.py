"""Assemble the Plot 1 building blocks into one publication figure.

The volume panels use broken linear y-axes and share one legend. The
representative inset helpers are retained, but their call is commented out.
All-seed comparisons remain separate
building blocks in plot1.py. The existing grids are rendered in memory
with larger labels and one shared legend for the eight 3-D panels. Only the
combined PDF is saved.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from matplotlib.ticker import FuncFormatter

import plot1


OUTPUT_PATH = plot1.OUTPUT_DIR / "plot1_combined.pdf"

PAGE_WIDTH = 1200.0
PAGE_HEIGHT = 900.0
PAGE_MARGIN = 8.0
COLUMN_GAP = 8.0
SUBPANEL_GAP = 4.0
TOP_PANEL_HEIGHT = 340.0
TOP_PANEL_MARGIN = 4.0
ROW_GAP = 5.0
PANEL_LABEL_HEIGHT = 18.0
VOLUME_LEGEND_HEIGHT = 36.0
GRID_LEGEND_HEIGHT = 36.0
DEVICE_BOX_LINEWIDTH = 1.0
LEGEND_FONTSIZE = 12
LEGEND_FONTFAMILY = "DejaVu Sans"

AXIS_LABELSIZE = 14
TICK_LABELSIZE = 11
DEVICE_LABELSIZE = 20
INSET_LABELSIZE = 10
INSET_TICK_LABELSIZE = 8
GRID_AXIS_LABELSIZE = 20
GRID_TICK_LABELSIZE = 15
SHOW_INSET_AXIS_LABELS = False
VOLUME_UPPER_LIMITS = (45, 60)
VOLUME_LOWER_LIMITS = (14, 40)
VOLUME_HEIGHT_RATIOS = (85, 15)

# Bounds are in parent-axis coordinates: left, bottom, width, height.
INSET_BOUNDS = (0.50, 0.10, 0.47, 0.46)

LEGEND_LABELS = (
    "PULSE Representative",
    "FLARE Representative",
    "Final volume (PULSE)",
    "Final volume (FLARE)",
    "Sobol limit",
)


def selected_seeds(
    device: str,
    pulse_results: dict[str, np.ndarray],
    flare_results: dict[str, np.ndarray],
) -> tuple[str, str, str, str]:
    """Apply the same reproducible seed selections as ``plot1.py``."""
    rng = np.random.default_rng(plot1.REPRESENTATIVE_RNG_SEEDS[device])
    pulse_seed = str(rng.choice(sorted(pulse_results)))
    flare_seed = str(rng.choice(sorted(flare_results)))
    if device == "H2-2":
        flare_seed = plot1.H2_2_FLARE_REPRESENTATIVE

    pulse_measurement_seed = str(
        np.random.default_rng(plot1.RNG_SEED).choice(sorted(pulse_results))
    )
    flare_cluster = [
        seed
        for seed in ("seed_2026", "seed_2028", "seed_2029")
        if seed in flare_results
    ]
    if not flare_cluster:
        flare_cluster = sorted(flare_results)
    flare_measurement_seed = str(
        np.random.default_rng(plot1.RNG_SEED).choice(flare_cluster)
    )
    return (
        pulse_seed,
        flare_seed,
        pulse_measurement_seed,
        flare_measurement_seed,
    )


def load_device_data(device: str) -> dict[str, object]:
    """Load existing per-seed and accumulated grids for one device."""
    pulse_root, flare_root, accumulated_root = plot1.VOLUME_ROOTS[device]
    pulse_results = plot1.load_seed_results(pulse_root, native=True)
    flare_results = plot1.load_seed_results(flare_root, native=False)
    pulse_grid = plot1.latest_accumulated_grid(
        accumulated_root,
        device,
        native=True,
    )
    flare_grid = plot1.latest_accumulated_grid(
        accumulated_root,
        device,
        native=False,
    )
    seeds = selected_seeds(device, pulse_results, flare_results)
    return {
        "pulse_results": pulse_results,
        "flare_results": flare_results,
        "pulse_final": plot1.success_volumes(pulse_grid, native=True),
        "flare_final": plot1.success_volumes(flare_grid),
        "pulse_seed": seeds[0],
        "flare_seed": seeds[1],
        "pulse_measurement_seed": seeds[2],
        "flare_measurement_seed": seeds[3],
    }


def style_inset(ax) -> None:
    """Apply the compact limits and typography used by the detail inset."""
    ax.set_xlim(150, 503)
    ax.set_ylim(48, 56)
    if SHOW_INSET_AXIS_LABELS:
        ax.set_xlabel("HQC spent", fontsize=INSET_LABELSIZE, labelpad=2)
        ax.set_ylabel("QARB-Volume", fontsize=INSET_LABELSIZE, labelpad=2)
    ax.tick_params(axis="both", labelsize=INSET_TICK_LABELSIZE, length=2)
    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.set_zorder(10)


def add_measurement_shot_axes(
    ax,
    pulse_rows: np.ndarray,
    flare_rows: np.ndarray,
    *,
    compact: bool = False,
) -> None:
    """Draw both shot-count rows as child axes with readable label sizes."""
    labelsize = 7 if compact else 10
    for rows, color, label_pad in (
        (pulse_rows, plot1.PULSE_COLOR, 2),
        (flare_rows, plot1.FLARE_COLOR, labelsize + 5),
    ):
        xmin, xmax = ax.get_xlim()
        visible_rows = rows[(rows[:, 0] >= xmin) & (rows[:, 0] <= xmax)]
        checkpoints = plot1.sparse_checkpoint_rows(
            visible_rows if len(visible_rows) else rows
        )
        # Figure-level twins can paint over a child inset's plotted data.
        top_ax = ax.secondary_xaxis("top")
        top_ax.set_xticks(checkpoints[:, 0])
        top_ax.set_xticklabels(
            [f"{int(shots):,}" for shots in checkpoints[:, 4]]
        )
        top_ax.tick_params(
            axis="x",
            colors=color,
            labelsize=labelsize,
            length=2,
            pad=label_pad,
        )
    if not compact or SHOW_INSET_AXIS_LABELS:
        top_ax.set_xlabel(
            "Measurement shots",
            fontsize=INSET_LABELSIZE if compact else AXIS_LABELSIZE,
            labelpad=4,
        )


def draw_representative_inset(
    parent_ax,
    data: dict[str, object],
) -> None:
    """Zoom in on the representative runs and use their own shot counts."""
    inset = parent_ax.inset_axes(INSET_BOUNDS)
    pulse_rows = data["pulse_results"][data["pulse_seed"]]
    flare_rows = data["flare_results"][data["flare_seed"]]

    inset.plot(
        pulse_rows[:, 0],
        pulse_rows[:, 1],
        color=plot1.PULSE_COLOR,
        marker="o",
        markersize=1.7,
        linewidth=0.8,
    )
    plot1.add_final_reference(
        inset,
        data["pulse_final"],
        color=plot1.PULSE_COLOR,
        method="PULSE",
        shade_bounds=True,
        bounds_alpha=0.25,
    )

    inset.plot(
        flare_rows[:, 0],
        flare_rows[:, 1],
        color=plot1.FLARE_COLOR,
        marker="o",
        markersize=1.7,
        linewidth=0.8,
    )
    plot1.add_final_reference(
        inset,
        data["flare_final"],
        color=plot1.FLARE_COLOR,
        method="FLARE",
        shade_bounds=True,
    )
    style_inset(inset)
    add_measurement_shot_axes(inset, pulse_rows, flare_rows, compact=True)


def draw_volume_panel(
    ax,
    data: dict[str, object],
    *,
    device: str,
    show_header: bool = True,
) -> None:
    """Draw unchanged trajectories for either section of the broken y-axis."""
    pulse_rows = data["pulse_results"][data["pulse_seed"]]
    flare_rows = data["flare_results"][data["flare_seed"]]

    ax.plot(
        pulse_rows[:, 0],
        pulse_rows[:, 1],
        color=plot1.PULSE_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="PULSE Representative",
    )
    ax.fill_between(
        pulse_rows[:, 0],
        pulse_rows[:, 2],
        pulse_rows[:, 3],
        color=plot1.PULSE_BAND_COLOR,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        flare_rows[:, 0],
        flare_rows[:, 1],
        color=plot1.FLARE_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="FLARE Representative",
    )
    ax.fill_between(
        flare_rows[:, 0],
        flare_rows[:, 2],
        flare_rows[:, 3],
        color=plot1.FLARE_BAND_COLOR,
        alpha=0.18,
        linewidth=0,
    )
    plot1.add_final_reference(
        ax,
        data["pulse_final"],
        color=plot1.PULSE_COLOR,
        method="PULSE",
        shade_bounds=True,
        bounds_alpha=0.25,
    )
    plot1.add_final_reference(
        ax,
        data["flare_final"],
        color=plot1.FLARE_COLOR,
        method="FLARE",
        shade_bounds=True,
    )
    ax.axvline(
        plot1.SOBOL_HQC_LIMIT,
        color=plot1.FLARE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label="Sobol limit",
    )

    ax.set_xlim(25, 501)
    ax.set_ylim((20, 60) if device == "H2-2" else (14, 60))
    ax.set_yscale("linear")
    ax.set_xlabel("HQC spent", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel("QARB-Volume", fontsize=AXIS_LABELSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABELSIZE)
    if show_header:
        ax.set_title(device, pad=2, fontsize=DEVICE_LABELSIZE)
        add_measurement_shot_axes(ax, pulse_rows, flare_rows)
    # draw_representative_inset(ax, data)


def volume_panels_pdf(
    device_data: dict[str, dict[str, object]],
) -> tuple[dict[str, bytes], bytes]:
    """Export each device with a compressed lower and enlarged upper y-axis."""
    panels = {}
    for device in ("H2-1", "H2-2"):
        fig, (upper_ax, lower_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8.0, 4.4),
            gridspec_kw={
                "height_ratios": VOLUME_HEIGHT_RATIOS,
                "hspace": 0.14,
            },
        )
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.78)
        draw_volume_panel(upper_ax, device_data[device], device=device)
        draw_volume_panel(
            lower_ax, device_data[device], device=device, show_header=False
        )
        upper_ax.set_ylim(*VOLUME_UPPER_LIMITS)
        lower_ax.set_ylim(*VOLUME_LOWER_LIMITS)
        upper_ax.set_ylim(45, 58)
        lower_ax.set_ylim(14, 30)
        # upper_ax.set_yticks([45, 50, 55])
        lower_ax.set_yticks([20, 30])
        upper_ax.set_xlabel("")
        upper_ax.set_ylabel("")
        lower_ax.set_ylabel("")
        fig.supylabel("QARB-Volume", x=0.04, y=0.45, fontsize=AXIS_LABELSIZE)
        upper_ax.spines["bottom"].set_visible(False)
        lower_ax.spines["top"].set_visible(False)
        upper_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Point-sized diagonal markers keep the break marks equal on both axes.
        for ax, edge in ((upper_ax, 0), (lower_ax, 1)):
            ax.plot(
                [0, 1],
                [edge, edge],
                transform=ax.transAxes,
                marker=[(-1, -0.5), (1, 0.5)],
                markersize=8,
                linestyle="none",
                color="black",
                markeredgewidth=1,
                clip_on=False,
                label="_nolegend_",
            )
        handles, labels = upper_ax.get_legend_handles_labels()
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf")
        panels[device] = buffer.getvalue()
        plt.close(fig)

    by_label = dict(zip(labels, handles))
    legend_content = shared_legend_pdf(
        [by_label[label] for label in LEGEND_LABELS],
        LEGEND_LABELS,
        handlelength=2.6,
    )
    return panels, legend_content


def shared_legend_pdf(
    handles,
    labels,
    *,
    handlelength: float,
) -> bytes:
    """Export a shared legend with a white background for clear overlays."""
    fig = plt.figure(figsize=(16.0, 0.5))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        prop={"family": LEGEND_FONTFAMILY, "size": LEGEND_FONTSIZE},
        handlelength=handlelength,
        columnspacing=1.5,
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", facecolor="white", transparent=False)
    plt.close(fig)
    return buffer.getvalue()


def fit_rect(source: pymupdf.Rect, target: pymupdf.Rect) -> pymupdf.Rect:
    """Center a PDF page inside a target rectangle without distortion."""
    scale = min(target.width / source.width, target.height / source.height)
    width = source.width * scale
    height = source.height * scale
    x0 = target.x0 + (target.width - width) / 2.0
    y0 = target.y0 + (target.height - height) / 2.0
    return pymupdf.Rect(x0, y0, x0 + width, y0 + height)


def place_pdf_bytes(
    page,
    target: pymupdf.Rect,
    content: bytes,
    *,
    trim_margins: bool = False,
) -> pymupdf.Rect:
    """Place an in-memory PDF page without rasterizing it."""
    with pymupdf.open(stream=content, filetype="pdf") as source:
        source_rect = (
            pdf_content_rect(source[0]) if trim_margins else source[0].rect
        )
        fitted = fit_rect(source_rect, target)
        page.show_pdf_page(fitted, source, 0, clip=source_rect)
    return fitted


def pdf_content_rect(page, padding: float = 4.0) -> pymupdf.Rect:
    """Locate visible ink; clip the original vector PDF rather than its preview."""
    pixmap = page.get_pixmap(colorspace=pymupdf.csRGB, alpha=False)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    rows, columns = np.nonzero(np.any(pixels < 250, axis=2))
    if not len(rows):
        return page.rect
    x_scale = page.rect.width / pixmap.width
    y_scale = page.rect.height / pixmap.height
    rect = pymupdf.Rect(
        page.rect.x0 + columns.min() * x_scale - padding,
        page.rect.y0 + rows.min() * y_scale - padding,
        page.rect.x0 + (columns.max() + 1) * x_scale + padding,
        page.rect.y0 + (rows.max() + 1) * y_scale + padding,
    )
    return rect & page.rect


def grid_panel_pdf(
    json_path: Path,
    grid_path: Path,
    *,
    name: str,
) -> tuple[bytes, dict[str, object]]:
    """Style a stored grid in memory and collect its real legend handles."""
    plotter = plot1.plot3d
    plotter.SHOW_ONE_SIGMA_SURFACES = False
    plotter.SHOW_TWO_SIGMA_SURFACES = True
    plotter.SHOW_MEASURED_POINTS = True
    native = name.startswith("pulse")
    method = "PULSE" if native else "FLARE"
    fig = plotter.plot_fle_isosurface_3d_matplotlib(
        json_path=json_path,
        grid_path=grid_path,
        xlim=(0.1, 1.0 if name == "flare_final" else 0.97),
        ylim=(20, 56),
        zlim=(2.0, 3.50),
        show_legend=False,
        axis_labelsize=GRID_AXIS_LABELSIZE,
        tick_labelsize=GRID_TICK_LABELSIZE,
        save=False,
        show=False,
    )
    ax = fig.axes[0]
    ax.set_zticks(np.log10([100, 300, 1000, 3000]))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{10**value:g}"))
    for label in fig.texts:
        if label.get_text() == r"$\log_{10}(n_{\rm g})$":
            label.set_text(r"$n_{\rm g}$")
    handles, labels = ax.get_legend_handles_labels()
    displayed_labels = {
        "Success-side volume": f"{method} success volume",
        (
            "PULSE posterior mean boundary"
            if native else "FLARE GP P(success) = 0.5"
        ): f"{method} levelset",
        "Failure": "Failure",
        "Success": "Success",
    }
    legend_handles = {
        displayed_labels[label]: handle
        for handle, label in zip(handles, labels)
        if label in displayed_labels
    }
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue(), legend_handles


def grid_legend_pdf(handles: dict[str, object]) -> bytes:
    """Create one shared grid legend from the plotted artists."""
    labels = [
        label
        for label in (
            "PULSE success volume", "PULSE levelset",
            "FLARE success volume", "FLARE levelset",
            "Failure", "Success",
        )
        if label in handles
    ]
    return shared_legend_pdf(
        [handles[label] for label in labels],
        labels,
        handlelength=2.4,
    )


def add_panel_label(page, rect: pymupdf.Rect, label: str) -> None:
    """Add one publication panel label in the source figure's white margin."""
    page.insert_text(
        (rect.x0 + 3.0, rect.y0 + 14.0),
        label,
        fontsize=12,
        fontname="Times-Roman",
        color=(0.0, 0.0, 0.0),
        overlay=True,
    )


def build_combined_figure(output_path: Path = OUTPUT_PATH) -> Path:
    """Create the final vector PDF using existing data and shared legends."""
    device_data = {
        device: load_device_data(device)
        for device in ("H2-1", "H2-2")
    }
    document = pymupdf.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    grid_legend_target = pymupdf.Rect(
        PAGE_MARGIN,
        PAGE_HEIGHT - PAGE_MARGIN - ROW_GAP - GRID_LEGEND_HEIGHT,
        PAGE_WIDTH - PAGE_MARGIN,
        PAGE_HEIGHT - PAGE_MARGIN - ROW_GAP,
    )
    volume_legend_target = pymupdf.Rect(
        PAGE_MARGIN,
        TOP_PANEL_HEIGHT + ROW_GAP / 2.0,
        PAGE_WIDTH - PAGE_MARGIN,
        TOP_PANEL_HEIGHT + ROW_GAP / 2.0 + VOLUME_LEGEND_HEIGHT,
    )
    device_box_bottom = PAGE_HEIGHT - PAGE_MARGIN

    volume_panels, volume_legend = volume_panels_pdf(device_data)

    column_width = (
        PAGE_WIDTH - 2.0 * PAGE_MARGIN - COLUMN_GAP
    ) / 2.0
    row_top = volume_legend_target.y1 + ROW_GAP
    row_height = (
        grid_legend_target.y0 - row_top - 2.0 * ROW_GAP
    ) / 2.0

    panel_labels = {
        "H2-1": ("a(2.1)", "a(2.2)", "a(3.1)", "a(3.2)"),
        "H2-2": ("b(2.1)", "b(2.2)", "b(3.1)", "b(3.2)"),
    }
    shared_grid_handles = {}
    for column, device in enumerate(("H2-1", "H2-2")):
        column_x = PAGE_MARGIN + column * (column_width + COLUMN_GAP)
        top_target = pymupdf.Rect(
            column_x + TOP_PANEL_MARGIN,
            PAGE_MARGIN,
            column_x + column_width - TOP_PANEL_MARGIN,
            TOP_PANEL_HEIGHT,
        )
        place_pdf_bytes(
            page, top_target, volume_panels[device], trim_margins=True
        )
        # Place all row labels in page coordinates so their left edges align.
        add_panel_label(
            page,
            pymupdf.Rect(
                column_x,
                top_target.y0 + PANEL_LABEL_HEIGHT,
                column_x + column_width,
                top_target.y1,
            ),
            f"{'ab'[column]}(1)",
        )
        subpanel_width = (column_width - SUBPANEL_GAP) / 2.0
        pulse_root, flare_root, _ = plot1.VOLUME_ROOTS[device]
        inputs = plot1.selected_grid_inputs(
            pulse_root,
            flare_root,
            device_data[device]["pulse_seed"],
            device_data[device]["flare_seed"],
        )

        for index, (name, label) in enumerate(
            zip(
                ("pulse_intermediate", "pulse_final", "flare_intermediate", "flare_final"),
                panel_labels[device],
            )
        ):
            row = index // 2
            subcolumn = index % 2
            x0 = column_x + subcolumn * (subpanel_width + SUBPANEL_GAP)
            y0 = row_top + row * (row_height + ROW_GAP)
            target = pymupdf.Rect(
                x0,
                y0,
                x0 + subpanel_width,
                y0 + row_height,
            )
            json_path, grid_path = inputs[name]
            content, legend_handles = grid_panel_pdf(
                json_path,
                grid_path,
                name=name,
            )
            shared_grid_handles.update(legend_handles)
            content_target = pymupdf.Rect(
                target.x0,
                target.y0 + PANEL_LABEL_HEIGHT,
                target.x1,
                target.y1,
            )
            place_pdf_bytes(page, content_target, content, trim_margins=True)
            add_panel_label(page, target, label)

    for column in range(2):
        x0 = PAGE_MARGIN + column * (column_width + COLUMN_GAP)
        page.draw_rect(
            pymupdf.Rect(
                x0,
                PAGE_MARGIN,
                x0 + column_width,
                device_box_bottom,
            ),
            color=(0.0, 0.0, 0.0),
            width=DEVICE_BOX_LINEWIDTH,
            overlay=True,
        )

    place_pdf_bytes(page, volume_legend_target, volume_legend)
    place_pdf_bytes(page, grid_legend_target, grid_legend_pdf(shared_grid_handles))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    if temporary_path.exists():
        temporary_path.unlink()
    document.save(temporary_path, garbage=4, deflate=True)
    document.close()
    temporary_path.replace(output_path)
    return output_path


def main() -> None:
    output_path = build_combined_figure()
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
