from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (
    BoxEditTool,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    RangeSlider,
    Slider,
)
from bokeh.plotting import figure

from kcwiulb.wcs import index_to_wavelength, wavelength_to_index


def _load_cube(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    path = Path(path)
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header.copy()
    return data, header


def _build_wavelength_array(header: fits.Header, n_wave: int) -> np.ndarray:
    return np.array([index_to_wavelength(i, header) for i in range(n_wave)], dtype=float)


def _build_collapse_mask(
    header: fits.Header,
    n_wave: int,
    collapse_exclude: tuple[float, float] | None,
) -> np.ndarray:
    mask = np.ones(n_wave, dtype=bool)

    if collapse_exclude is not None:
        i0 = max(0, wavelength_to_index(collapse_exclude[0], header))
        i1 = min(n_wave, wavelength_to_index(collapse_exclude[1], header))
        if i1 > i0:
            mask[i0:i1] = False

    return mask


def _collapsed_image(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.sum(cube[mask], axis=0)


def create_interactive_viewer(
    flux_path: str | Path,
    residual_path: str | Path,
    model_path: str | Path,
    comparison_path: str | Path | None = None,
    collapse_exclude: tuple[float, float] | None = None,
    spectrum_x_range: tuple[float, float] | None = None,
    image_low: float = -0.001,
    image_high: float = 0.001,
    collapse_low: float = -0.1,
    collapse_high: float = 0.3,
    height1: int = 500,
    height2: int = 300,
    width2: int = 800,
    doc=None,
) -> None:

    """
    Create a Bokeh app for interactive cube inspection.
    """
    if doc is None:
        doc = curdoc()

    flux_cube, header = _load_cube(flux_path)
    residual_cube, _ = _load_cube(residual_path)
    model_cube, _ = _load_cube(model_path)

    if flux_cube.shape != residual_cube.shape or flux_cube.shape != model_cube.shape:
        raise ValueError("flux, residual, and model cubes must have the same shape")

    comparison_cube = None
    if comparison_path is not None:
        comparison_cube, _ = _load_cube(comparison_path)
        if comparison_cube.shape != flux_cube.shape:
            raise ValueError("comparison cube must have the same shape as the main cubes")

    n_wave, ny, nx = flux_cube.shape

    # Image panels: preserve true FOV shape
    width1 = int(height1 * nx / ny)

    # Spectrum panel: free width from wrapper

    wl = _build_wavelength_array(header, n_wave)

    if spectrum_x_range is None:
        spectrum_x_range = (float(wl[0]), float(wl[-1]))

    collapse_mask = _build_collapse_mask(header, n_wave, collapse_exclude)
    flux_collapsed = _collapsed_image(flux_cube, collapse_mask)

    if comparison_cube is not None:
        comparison_collapsed = _collapsed_image(comparison_cube, collapse_mask)
    else:
        comparison_collapsed = None

    color_mapper = LinearColorMapper(
        palette="Turbo256",
        low=image_low,
        high=image_high,
    )
    color_mapper2 = LinearColorMapper(
        palette="Turbo256",
        low=collapse_low,
        high=collapse_high,
    )

    source_flux_image = ColumnDataSource(data=dict(image=[flux_cube[0]]))
    source_resid_image = ColumnDataSource(data=dict(image=[residual_cube[0]]))
    source_flux_spec = ColumnDataSource(data=dict(x=wl, y=flux_cube[:, 0, 0]))
    source_resid_spec = ColumnDataSource(data=dict(x=wl, y=residual_cube[:, 0, 0]))
    source_model_spec = ColumnDataSource(data=dict(x=wl, y=model_cube[:, 0, 0]))
    source_flux_collapse = ColumnDataSource(data=dict(image=[flux_collapsed]))

    if comparison_collapsed is not None:
        source_comp_collapse = ColumnDataSource(data=dict(image=[comparison_collapsed]))
    else:
        source_comp_collapse = None

    roi_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

    p1 = figure(
        title="Original Cube",
        x_range=(0, nx),
        y_range=(0, ny),
        match_aspect=True,
        height=height1,
        width=width1,
    )
    p2 = figure(
        title="Residual / Subtracted Cube",
        x_range=(0, nx),
        y_range=(0, ny),
        match_aspect=True,
        height=height1,
        width=width1,
    )
    p3 = figure(
        title="Spectrum",
        x_range=spectrum_x_range,
        x_axis_label="Wavelength (Angstrom)",
        y_axis_label="Flux",
        height=height2,
        width=width2,
    )
    p5 = figure(
        title="Collapsed Image",
        x_range=(0, nx),
        y_range=(0, ny),
        match_aspect=True,
        height=height1,
        width=width1,
    )

    if comparison_cube is not None:
        p7 = figure(
            title="Comparison Collapsed Image",
            x_range=(0, nx),
            y_range=(0, ny),
            match_aspect=True,
            height=height1,
            width=width1,
        )
    else:
        p7 = None

    p1.image(
        source=source_flux_image,
        image="image",
        x=0,
        y=0,
        dw=nx,
        dh=ny,
        color_mapper=color_mapper,
    )
    p2.image(
        source=source_resid_image,
        image="image",
        x=0,
        y=0,
        dw=nx,
        dh=ny,
        color_mapper=color_mapper,
    )
    p3.line(
        "x", "y",
        source=source_flux_spec,
        legend_label="Original",
        line_color="blue",
        line_width=2,
    )
    p3.line(
        "x", "y",
        source=source_model_spec,
        legend_label="Model",
        line_color="green",
        line_width=2,
    )
    p3.line(
        "x", "y",
        source=source_resid_spec,
        legend_label="Residual",
        line_color="red",
        line_width=2,
    )
    p5.image(
        source=source_flux_collapse,
        image="image",
        x=0,
        y=0,
        dw=nx,
        dh=ny,
        color_mapper=color_mapper2,
    )

    if p7 is not None and source_comp_collapse is not None:
        p7.image(
            source=source_comp_collapse,
            image="image",
            x=0,
            y=0,
            dw=nx,
            dh=ny,
            color_mapper=color_mapper2,
        )

    wav_slider = Slider(
        start=float(wl[0]),
        end=float(wl[-1]),
        value=float(wl[0]),
        step=float(header["CD3_3"]),
        title="Wavelength",
    )

    def wl_to_index_local(w: float) -> int:
        return int(np.clip(wavelength_to_index(w, header), 0, n_wave - 1))

    def update_wavelength_slice(attr, old, new):
        idx = wl_to_index_local(wav_slider.value)
        source_flux_image.data = dict(image=[flux_cube[idx]])
        source_resid_image.data = dict(image=[residual_cube[idx]])

    wav_slider.on_change("value", update_wavelength_slice)

    range_slider = RangeSlider(
        title="Adjust x-axis range",
        start=float(wl[0]),
        end=float(wl[-1]),
        step=1,
        value=spectrum_x_range,
    )
    range_slider.js_link("value", p3.x_range, "start", attr_selector=0)
    range_slider.js_link("value", p3.x_range, "end", attr_selector=1)

    def update_spectra_from_region(x0: int, x1: int, y0: int, y1: int) -> None:
        x0 = max(0, min(nx - 1, x0))
        x1 = max(0, min(nx - 1, x1))
        y0 = max(0, min(ny - 1, y0))
        y1 = max(0, min(ny - 1, y1))

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        flux_spec = np.median(flux_cube[:, y0:y1 + 1, x0:x1 + 1], axis=(1, 2))
        resid_spec = np.median(residual_cube[:, y0:y1 + 1, x0:x1 + 1], axis=(1, 2))
        model_spec = np.median(model_cube[:, y0:y1 + 1, x0:x1 + 1], axis=(1, 2))

        source_flux_spec.data = dict(x=wl, y=flux_spec)
        source_resid_spec.data = dict(x=wl, y=resid_spec)
        source_model_spec.data = dict(x=wl, y=model_spec)

    def hover_spectra(event):
        if len(roi_source.data["width"]) > 0 and len(roi_source.data["height"]) > 0:
            width = roi_source.data["width"][0]
            height = roi_source.data["height"][0]
        else:
            width = 0
            height = 0

        x_start = int(event.x - 0.5 * width)
        x_end = int(event.x + 0.5 * width)
        y_start = int(event.y - 0.5 * height)
        y_end = int(event.y + 0.5 * height)

        update_spectra_from_region(x_start, x_end, y_start, y_end)

    def tap_spectra(event):
        x = int(event.x)
        y = int(event.y)
        update_spectra_from_region(x, x, y, y)

    def box_median(attr, old, new):
        rois = roi_source.data
        if len(rois["x"]) == 0:
            return

        x_center = rois["x"][0]
        y_center = rois["y"][0]
        width = rois["width"][0]
        height = rois["height"][0]

        x_start = int(x_center - 0.5 * width)
        x_end = int(x_center + 0.5 * width)
        y_start = int(y_center - 0.5 * height)
        y_end = int(y_center + 0.5 * height)

        update_spectra_from_region(x_start, x_end, y_start, y_end)

    p1.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))
    p2.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))
    p3.add_tools(HoverTool(tooltips=[("(x,y)", "($x, $y)")]))
    p5.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))
    if p7 is not None:
        p7.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    color_bar2 = ColorBar(color_mapper=color_mapper2, label_standoff=12)

    panels = [p1, p2, p5]
    if p7 is not None:
        panels.append(p7)

    for p in panels:
        p.grid.grid_line_width = 0

    p1.add_layout(color_bar, "below")
    p2.add_layout(color_bar, "below")
    p5.add_layout(color_bar2, "below")
    if p7 is not None:
        p7.add_layout(color_bar2, "below")

    for p in panels:
        p.on_event(events.MouseMove, hover_spectra)
        p.on_event("tap", tap_spectra)

    renderer1 = p1.rect(
        "x", "y", "width", "height",
        source=roi_source,
        alpha=0.5,
        fill_color=None,
        line_alpha=1,
        line_color="black",
    )
    renderer2 = p2.rect(
        "x", "y", "width", "height",
        source=roi_source,
        alpha=0.5,
        fill_color=None,
        line_alpha=1,
        line_color="black",
    )
    renderer5 = p5.rect(
        "x", "y", "width", "height",
        source=roi_source,
        alpha=0.5,
        fill_color=None,
        line_alpha=1,
        line_color="black",
    )

    draw_tool1 = BoxEditTool(renderers=[renderer1], num_objects=1)
    draw_tool2 = BoxEditTool(renderers=[renderer2], num_objects=1)
    draw_tool5 = BoxEditTool(renderers=[renderer5], num_objects=1)

    p1.add_tools(draw_tool1)
    p2.add_tools(draw_tool2)
    p5.add_tools(draw_tool5)

    p1.toolbar.active_drag = draw_tool1
    p2.toolbar.active_drag = draw_tool2
    p5.toolbar.active_drag = draw_tool5

    if p7 is not None:
        renderer7 = p7.rect(
            "x", "y", "width", "height",
            source=roi_source,
            alpha=0.5,
            fill_color=None,
            line_alpha=1,
            line_color="black",
        )
        draw_tool7 = BoxEditTool(renderers=[renderer7], num_objects=1)
        p7.add_tools(draw_tool7)
        p7.toolbar.active_drag = draw_tool7

    roi_source.on_change("data", box_median)

    if p7 is not None:
        image_rows = [
            [p1, p2],
            [p5, p7],
        ]
    else:
        image_rows = [
            [p1, p2],
            [p5],
        ]

    app_layout = layout(
        [
            [wav_slider],
            *image_rows,
            [range_slider],
            [p3],
        ]
    )

    doc.add_root(app_layout)
    doc.title = "Interactive Cube Viewer"