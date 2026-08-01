"""Reusable helpers that turn FEM nodal field data into self-contained,
polished, interactive HTML fragments for a study's ``viz/<name>.html`` page
in the read-only vivarium-workbench.

Every function returns a single ``str`` fragment (a ``<div>`` + inline
``<script>``, CDN Plotly or Three.js) meant to be embedded directly into a
larger page -- no ``<html>``/``<head>``/``<body>`` wrapper of its own.

Colors follow the pbg-superpowers ``dataviz`` skill's validated default
palette (see ``pbg-superpowers/.../skills/dataviz/references/palette.md``):

- **Sequential** magnitude encoding (field value, speed, mesh value) always
  uses the single blue ramp (steps 100->700, light->dark, 13 stops) -- the
  same hex values in both light and dark mode, since a sequential ramp
  doesn't re-theme, only chart chrome (background/text/axes) does.
- **Categorical** identity encoding (the two series in the convergence plot,
  the quiver direction line) uses slots 1/2 (blue/orange) in the palette's
  fixed order; unlike the sequential ramp, light and dark each have their
  own validated step, so these marks *do* re-theme (via `meta="series-N"`
  tags the post_script restyles on toggle, same as chart chrome).
- Chart chrome (background, text, axis, grid, colorbar) is theme-aware: each
  Plotly fragment ships a small ``post_script`` that reads the host page's
  ``data-theme`` attribute (falling back to ``prefers-color-scheme``), applies
  the matching light/dark chrome via ``Plotly.relayout``, and re-applies it
  live via a ``MutationObserver`` + media-query listener, matching the
  pattern documented in the dataviz skill's ``palette.md``. The Three.js
  fragment uses the same CSS-custom-property pattern for its HTML/CSS
  overlay (title, colorbar legend, slider) and needs no JS re-theme since
  its canvas is alpha-transparent over the themed card background.

Mesh handling
-------------
Fields live on an unstructured triangular mesh (dolfinx's "crossed"
diagonal unit-square triangulation is deliberately NOT a regular grid --
see ``fem.build_mesh``). ``field_heatmap_html`` / ``field_animation_html``
only need ``(coords, values)``: they interpolate nodal values onto a
regular grid with ``scipy.interpolate.griddata`` (linear, NaN outside the
convex hull) and render a Plotly ``Contour`` filled surface -- robust to
any node layout, no cell connectivity required. ``mesh3d_html`` is the one
function that *does* want connectivity (``cells``, triangle vertex
indices) to build a true triangulated Three.js ``BufferGeometry`` surface.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# CDN pins
# ---------------------------------------------------------------------------

# NOTE: pinned to the plotly.js release that matches the installed plotly.py
# (verified via `plotly.offline.offline.get_plotlyjs_version()`), NOT an
# arbitrary version string. plotly.py >=6 serializes numpy arrays as compact
# typed-array JSON (`{"dtype": ..., "bdata": ..., "shape": ...}`) instead of
# plain lists; older plotly.js builds (e.g. 2.27.0) don't know how to decode
# that format and silently render blank traces / broken axis ranges -- this
# was caught by screenshotting a rendered fragment in a headless browser
# during development, not by the unit tests (which only check for string
# markers, not that Plotly.js actually painted the trace).
PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.7.0.min.js"
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
THREE_ORBIT_CDN = (
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
)

_FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# dataviz skill palette.md: categorical slots 1 (blue) / 2 (orange) / 3 (aqua)
# / 4 (yellow), fixed order -- used for identity marks (the convergence
# plot's two series, the quiver direction line). Light and dark each have
# their own validated steps (see palette.md's categorical table); traces
# using these are tagged with `meta="series-N"` so the post_script's theme
# re-application can restyle them to the matching-mode step on toggle.
SERIES = SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

# dataviz skill palette.md: sequential blue ramp, steps 100->700, 13 stops,
# light->dark. Same hex values regardless of chart theme -- used for every
# magnitude encoding (field value, speed, mesh value).
_SEQ_STOPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def _seq_colorscale():
    """Plotly colorscale (list of [pos, hex]) for the dataviz sequential ramp."""
    n = len(_SEQ_STOPS) - 1
    return [[i / n, c] for i, c in enumerate(_SEQ_STOPS)]


def _new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Card chrome (shared wrapper, dataviz palette.md CSS-custom-property pattern)
# ---------------------------------------------------------------------------

_CARD_STYLE = """
<style>
.pbg-viz-card {
  --pbg-surface: #fcfcfb;
  --pbg-page: #f9f9f7;
  --pbg-text: #0b0b0b;
  --pbg-text-secondary: #52514e;
  --pbg-muted: #898781;
  --pbg-border: rgba(11,11,11,0.10);
  color-scheme: light;
  background: var(--pbg-surface);
  border: 1px solid var(--pbg-border);
  border-radius: 12px;
  padding: 14px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .pbg-viz-card {
    --pbg-surface: #1a1a19;
    --pbg-page: #0d0d0d;
    --pbg-text: #ffffff;
    --pbg-text-secondary: #c3c2b7;
    --pbg-muted: #898781;
    --pbg-border: rgba(255,255,255,0.10);
    color-scheme: dark;
  }
}
:root[data-theme="dark"] .pbg-viz-card {
  --pbg-surface: #1a1a19;
  --pbg-page: #0d0d0d;
  --pbg-text: #ffffff;
  --pbg-text-secondary: #c3c2b7;
  --pbg-muted: #898781;
  --pbg-border: rgba(255,255,255,0.10);
  color-scheme: dark;
}
</style>
"""


def _card_open(extra_style=""):
    return f'<div class="pbg-viz-card">{_CARD_STYLE}{extra_style}'


def _card_close():
    return "</div>"


# ---------------------------------------------------------------------------
# Theme-aware relayout for Plotly fragments (light/dark chart chrome)
# ---------------------------------------------------------------------------

# Inserted verbatim as the *value* substituted into plotly's own post_script
# template (a simple str.replace, not str.format) -- safe to contain any
# number of literal JS braces. `{plot_id}` is plotly's own placeholder token,
# replaced with the actual div id by `fig.to_html`.
_THEME_POST_SCRIPT = """
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd) return;
  function theme() {
    var attr = document.documentElement.getAttribute('data-theme');
    var dark = attr ? attr === 'dark' : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    return dark ? {
      text: '#ffffff', textSecondary: '#c3c2b7', muted: '#898781',
      grid: '#2c2c2a', axis: '#383835', surface: '#1a1a19',
      series: ['#3987e5', '#d95926', '#199e70', '#c98500']
    } : {
      text: '#0b0b0b', textSecondary: '#52514e', muted: '#898781',
      grid: '#e1e0d9', axis: '#c3c2b7', surface: '#fcfcfb',
      series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    };
  }
  function apply() {
    var t = theme();
    var patch = {
      'paper_bgcolor': 'rgba(0,0,0,0)',
      'plot_bgcolor': 'rgba(0,0,0,0)',
      'font.color': t.textSecondary,
      'title.font.color': t.text,
      'legend.font.color': t.textSecondary,
      'xaxis.gridcolor': t.grid, 'xaxis.linecolor': t.axis, 'xaxis.zerolinecolor': t.axis,
      'xaxis.tickfont.color': t.muted, 'xaxis.title.font.color': t.textSecondary,
      'yaxis.gridcolor': t.grid, 'yaxis.linecolor': t.axis, 'yaxis.zerolinecolor': t.axis,
      'yaxis.tickfont.color': t.muted, 'yaxis.title.font.color': t.textSecondary,
      'coloraxis.colorbar.tickfont.color': t.muted,
      'coloraxis.colorbar.title.font.color': t.textSecondary,
      'coloraxis.colorbar.outlinewidth': 0
    };
    if (gd.layout && gd.layout.annotations) {
      for (var i = 0; i < gd.layout.annotations.length; i++) {
        patch['annotations[' + i + '].font.color'] = t.textSecondary;
      }
    }
    if (gd.layout && gd.layout.sliders && gd.layout.sliders.length) {
      patch['sliders[0].font.color'] = t.muted;
      patch['sliders[0].currentvalue.font.color'] = t.textSecondary;
      patch['sliders[0].bgcolor'] = t.surface;
      patch['sliders[0].bordercolor'] = t.axis;
      patch['sliders[0].activebgcolor'] = t.axis;
    }
    if (gd.layout && gd.layout.updatemenus && gd.layout.updatemenus.length) {
      patch['updatemenus[0].bgcolor'] = t.surface;
      patch['updatemenus[0].bordercolor'] = t.axis;
      patch['updatemenus[0].font.color'] = t.textSecondary;
    }
    Plotly.relayout(gd, patch);
    if (gd.data) {
      for (var k = 0; k < gd.data.length; k++) {
        if (gd.data[k].marker && gd.data[k].marker.line) {
          Plotly.restyle(gd, {'marker.line.color': t.surface}, [k]);
        }
        var meta = gd.data[k].meta;
        if (typeof meta === 'string' && meta.indexOf('series-') === 0) {
          var seriesIdx = parseInt(meta.split('-')[1], 10) - 1;
          var color = t.series[seriesIdx % t.series.length];
          Plotly.restyle(gd, {'line.color': color, 'marker.color': color}, [k]);
        }
      }
    }
  }
  apply();
  var mo = new MutationObserver(apply);
  mo.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) { mq.addEventListener('change', apply); }
    else if (mq.addListener) { mq.addListener(apply); }
  }
})();
"""

_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


def _finish(fig, height, div_id=None):
    """Render a themed go.Figure to a self-contained CDN-Plotly card fragment."""
    div_id = div_id or _new_id("pbg-viz")
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT_FAMILY, size=12, color="#52514e"),
        hovermode="closest",
        hoverlabel=dict(font=dict(family=_FONT_FAMILY, size=12)),
    )
    plot_html = fig.to_html(
        config=_PLOTLY_CONFIG,
        include_plotlyjs=PLOTLY_CDN,
        full_html=False,
        div_id=div_id,
        post_script=_THEME_POST_SCRIPT,
        default_width="100%",
        default_height=f"{height}px",
    )
    return _card_open() + plot_html + _card_close()


# ---------------------------------------------------------------------------
# Grid interpolation (mesh-agnostic: only needs (coords, values))
# ---------------------------------------------------------------------------

def _interpolate_to_grid(coords, values, n=140):
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    x, y = coords[:, 0], coords[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin:
        xmax = xmin + 1e-9
    if ymax <= ymin:
        ymax = ymin + 1e-9
    xi = np.linspace(xmin, xmax, n)
    yi = np.linspace(ymin, ymax, n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata(coords, values, (XI, YI), method="linear")
    return xi, yi, ZI


# ---------------------------------------------------------------------------
# 1. field_heatmap_html
# ---------------------------------------------------------------------------

def field_heatmap_html(coords, values, title):
    """Triangulated 2D scalar field (interpolated contour/heatmap), colorbar."""
    xi, yi, zi = _interpolate_to_grid(coords, values)
    fig = go.Figure(
        data=go.Contour(
            x=xi,
            y=yi,
            z=zi,
            coloraxis="coloraxis",
            contours=dict(coloring="heatmap", showlines=False),
            connectgaps=False,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>value=%{z:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        coloraxis=dict(
            colorscale=_seq_colorscale(),
            colorbar=dict(title=dict(text="value", font=dict(color="#52514e")), outlinewidth=0),
        ),
        xaxis=dict(title=dict(text="x", font=dict(color="#52514e")), scaleanchor="y", constrain="domain"),
        yaxis=dict(title=dict(text="y", font=dict(color="#52514e"))),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return _finish(fig, height=480)


# ---------------------------------------------------------------------------
# 2. field_animation_html
# ---------------------------------------------------------------------------

def field_animation_html(coords, frames, times, title, n=90):
    """Plotly animation of a scalar field over time, with a slider + play/pause.

    Viz-size budget: an animation embeds ``n*n`` JSON-encoded floats PER
    FRAME (every frame's full interpolated grid, not just a diff), so file
    size grows with ``n**2 * len(frames)`` -- easy to blow past several MB.
    ``n=90`` (vs. the single-frame ``field_heatmap_html``'s ``n=140``) still
    comfortably exceeds a typical FEM mesh's node resolution while keeping
    per-frame payload down. Target: animated viz <=~30 frames, ``n<=~100``,
    <=~6MB per file (a study with a longer/finer run should subsample
    frames -- e.g. every Nth snapshot -- rather than growing ``n`` or frame
    count unboundedly; see ``studies/navier-stokes/sims/run.py``'s
    ``SNAPSHOT_DT`` for the frame-cadence knob).
    """
    frames = [np.asarray(f, dtype=float) for f in frames]
    times = list(times)

    xi = yi = None
    zis = []
    for f in frames:
        xi, yi, zi = _interpolate_to_grid(coords, f, n=n)
        zis.append(zi)

    vmin = float(np.nanmin([np.nanmin(z) for z in zis]))
    vmax = float(np.nanmax([np.nanmax(z) for z in zis]))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    def _contour(zi):
        return go.Contour(
            x=xi,
            y=yi,
            z=zi,
            coloraxis="coloraxis",
            contours=dict(coloring="heatmap", showlines=False),
            connectgaps=False,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>value=%{z:.4g}<extra></extra>",
        )

    frame_names = [f"{t:.6g}" for t in times]
    fig_frames = [go.Frame(data=[_contour(zi)], name=name) for zi, name in zip(zis, frame_names)]
    fig = go.Figure(data=[_contour(zis[0])], frames=fig_frames)

    slider_steps = [
        dict(
            label=f"{t:.4g}",
            method="animate",
            args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        )
        for t, name in zip(times, frame_names)
    ]

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        coloraxis=dict(
            colorscale=_seq_colorscale(),
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=dict(text="value", font=dict(color="#52514e")), outlinewidth=0),
        ),
        xaxis=dict(title=dict(text="x", font=dict(color="#52514e")), scaleanchor="y", constrain="domain"),
        yaxis=dict(title=dict(text="y", font=dict(color="#52514e"))),
        margin=dict(l=60, r=30, t=50, b=90),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=-0.18,
                xanchor="left",
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=400, redraw=True), fromcurrent=True, transition=dict(duration=0)),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.02,
                len=0.94,
                pad=dict(t=40),
                currentvalue=dict(prefix="t = ", font=dict(color="#52514e")),
                font=dict(color="#898781"),
                steps=slider_steps,
            )
        ],
    )
    return _finish(fig, height=560)


# ---------------------------------------------------------------------------
# 2a. field_animation_with_wall_envelope_html
# ---------------------------------------------------------------------------

def field_animation_with_wall_envelope_html(
    coords, frames, times, x_wall, wall_top_frames, wall_bottom_frames, title, n=90,
):
    """``field_animation_html`` (see its docstring for the grid-interpolation
    / size-budget mechanics) PLUS two per-frame line traces overlaying the
    TRUE (deformed) top/bottom wall boundary -- for a moving-mesh study
    where the field animation itself must stay at FIXED reference node
    positions (``field_animation_html`` has no per-frame-coordinate
    support; see its docstring), so the wall-envelope overlay is the only
    part of the animation that visibly moves, honestly conveying the real
    traveling-wave deformation the color-encoded field values were computed
    on top of (see ``studies/moving-boundary/sims/run.py``'s viz docstring
    for the full caveat).

    Args:
        coords: (N, 2) REFERENCE dof coordinates the field is plotted at
            (shared across all frames, same convention as
            ``field_animation_html``).
        frames: sequence of (N,) field-value arrays, one per frame.
        times: sequence of time labels, one per frame.
        x_wall: (M,) shared x-sample coordinates for the wall curves.
        wall_top_frames, wall_bottom_frames: sequence of (M,) y-coordinate
            arrays (one per frame) giving the TRUE current top/bottom wall
            position at each ``x_wall`` sample.
        title: card title.
        n: interpolation grid resolution (see ``field_animation_html``).
    """
    frames = [np.asarray(f, dtype=float) for f in frames]
    times = list(times)
    x_wall = np.asarray(x_wall, dtype=float)

    xi = yi = None
    zis = []
    for f in frames:
        xi, yi, zi = _interpolate_to_grid(coords, f, n=n)
        zis.append(zi)

    vmin = float(np.nanmin([np.nanmin(z) for z in zis]))
    vmax = float(np.nanmax([np.nanmax(z) for z in zis]))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    def _contour(zi):
        return go.Contour(
            x=xi, y=yi, z=zi, coloraxis="coloraxis",
            contours=dict(coloring="heatmap", showlines=False),
            connectgaps=False,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>value=%{z:.4g}<extra></extra>",
        )

    def _wall_trace(y_vals, name, color):
        return go.Scatter(
            x=x_wall, y=np.asarray(y_vals, dtype=float), mode="lines", meta="series-1",
            line=dict(width=3, color=color), hoverinfo="skip", showlegend=False, name=name,
        )

    WALL_COLOR = "#0b0b0b"
    frame_names = [f"{t:.6g}" for t in times]
    fig_frames = [
        go.Frame(
            data=[_contour(zi), _wall_trace(top, "wall (top)", WALL_COLOR), _wall_trace(bottom, "wall (bottom)", WALL_COLOR)],
            name=name,
        )
        for zi, top, bottom, name in zip(zis, wall_top_frames, wall_bottom_frames, frame_names)
    ]
    fig = go.Figure(
        data=[_contour(zis[0]), _wall_trace(wall_top_frames[0], "wall (top)", WALL_COLOR),
              _wall_trace(wall_bottom_frames[0], "wall (bottom)", WALL_COLOR)],
        frames=fig_frames,
    )

    slider_steps = [
        dict(
            label=f"{t:.4g}", method="animate",
            args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        )
        for t, name in zip(times, frame_names)
    ]

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        coloraxis=dict(
            colorscale=_seq_colorscale(), cmin=vmin, cmax=vmax,
            colorbar=dict(title=dict(text="value", font=dict(color="#52514e")), outlinewidth=0),
        ),
        xaxis=dict(title=dict(text="x", font=dict(color="#52514e")), scaleanchor="y", constrain="domain"),
        yaxis=dict(title=dict(text="y", font=dict(color="#52514e"))),
        margin=dict(l=60, r=30, t=50, b=90),
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.02, y=-0.18, xanchor="left", yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play", method="animate",
                        args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True, transition=dict(duration=0))],
                    ),
                    dict(
                        label="⏸ Pause", method="animate",
                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0, x=0.02, len=0.94, pad=dict(t=40),
                currentvalue=dict(prefix="t = ", font=dict(color="#52514e")),
                font=dict(color="#898781"), steps=slider_steps,
            )
        ],
    )
    return _finish(fig, height=560)


# ---------------------------------------------------------------------------
# 2b. mesh_wireframe_animation_html
# ---------------------------------------------------------------------------

def mesh_wireframe_animation_html(frames_coords, frames_cells, labels, title, highlight=None):
    """Animated triangle-mesh WIREFRAME (edges only, no field data) across a
    sequence of DIFFERENT meshes -- e.g. successive AMR levels -- so element
    density/concentration is directly visible, unlike ``field_animation_html``
    (which needs a scalar field and re-interpolates every frame onto a
    fixed-size grid).

    Each frame is its own independent mesh (different node count / different
    triangle count from every other frame -- NOT a fixed mesh with a
    time-varying field), rendered as a single ``Scatter`` "lines" trace per
    frame: every triangle's 3 edges drawn as (x0,x1,None)/(y0,y1,None)
    segments (mirrors ``quiver_streamlines_html``'s ``None``-separated
    multi-segment-line trick). No grid interpolation, no field values -- just
    real mesh geometry, so this works for ANY triangulation regardless of
    what (if anything) is solved on it.

    Viz-size budget: unlike ``field_animation_html`` (whose payload is
    ``O(n^2 * n_frames)`` from grid interpolation), this is
    ``O(cells_in_frame)`` per frame, summed over frames -- coordinates are
    rounded to 5 significant figures before JSON-encoding to keep the
    (x,y,None)-per-edge payload compact. Callers with meshes that grow into
    the tens of thousands of cells per frame should cap ``frames_cells`` to
    the early/illustrative levels of a refinement sequence (a full AMR run's
    LATER, much finer levels are better conveyed via the scalar
    convergence chart than via full mesh geometry) -- see
    ``studies/mesh-convergence/sims/run.py``'s ``ANIMATION_LEVELS``.

    Args:
        frames_coords: sequence of (N_i, 2) arrays, one per frame (node
            coordinates -- N_i varies frame to frame).
        frames_cells: sequence of (M_i, 3) int arrays, one per frame
            (triangle vertex-index connectivity into the matching
            ``frames_coords[i]``).
        labels: sequence of str, one per frame (slider step label, e.g.
            "level 3: 358 cells").
        title: card title.
        highlight: optional (x, y) point to mark with a small red dot in
            every frame (e.g. the re-entrant corner being refined toward).
    """
    labels = list(labels)
    n_frames = len(labels)

    def _edges_xy(coords, cells):
        coords = np.asarray(coords, dtype=float)
        cells = np.asarray(cells, dtype=int)
        xs, ys = [], []
        for tri in cells:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                x0, y0 = coords[tri[a]]
                x1, y1 = coords[tri[b]]
                xs += [round(float(x0), 5), round(float(x1), 5), None]
                ys += [round(float(y0), 5), round(float(y1), 5), None]
        return xs, ys

    frame_edges = [
        _edges_xy(frames_coords[i], frames_cells[i]) for i in range(n_frames)
    ]

    all_coords = np.vstack([np.asarray(c, dtype=float) for c in frames_coords])
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = 0.04 * (xmax - xmin or 1.0)
    pad_y = 0.04 * (ymax - ymin or 1.0)

    def _mesh_trace(xs, ys):
        return go.Scatter(
            x=xs, y=ys, mode="lines", meta="series-1",
            line=dict(width=1, color=SERIES[0]), opacity=0.75,
            hoverinfo="skip", showlegend=False, name="mesh",
        )

    fig_frames = [
        go.Frame(data=[_mesh_trace(xs, ys)], name=labels[i])
        for i, (xs, ys) in enumerate(frame_edges)
    ]
    xs0, ys0 = frame_edges[0]
    data = [_mesh_trace(xs0, ys0)]
    if highlight is not None:
        data.append(
            go.Scatter(
                x=[highlight[0]], y=[highlight[1]], mode="markers",
                marker=dict(size=9, color="#c0392b", line=dict(width=1, color="#fcfcfb")),
                hovertemplate="re-entrant corner<extra></extra>",
                showlegend=False, name="corner",
            )
        )

    fig = go.Figure(data=data, frames=fig_frames)

    slider_steps = [
        dict(
            label=labels[i], method="animate",
            args=[[labels[i]], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        )
        for i in range(n_frames)
    ]

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(
            title=dict(text="x", font=dict(color="#52514e")),
            range=[xmin - pad_x, xmax + pad_x],
            scaleanchor="y", constrain="domain",
        ),
        yaxis=dict(title=dict(text="y", font=dict(color="#52514e")), range=[ymin - pad_y, ymax + pad_y]),
        margin=dict(l=60, r=30, t=50, b=90),
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.02, y=-0.18, xanchor="left", yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play", method="animate",
                        args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True, transition=dict(duration=0))],
                    ),
                    dict(
                        label="⏸ Pause", method="animate",
                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0, x=0.02, len=0.94, pad=dict(t=40),
                currentvalue=dict(prefix="", font=dict(color="#52514e")),
                font=dict(color="#898781"), steps=slider_steps,
            )
        ],
    )
    return _finish(fig, height=560)


# ---------------------------------------------------------------------------
# 3. convergence_loglog_html
# ---------------------------------------------------------------------------

def convergence_loglog_html(h, errors):
    """Log-log error-vs-mesh-size plot with a fitted-slope (convergence rate) annotation."""
    h = np.asarray(h, dtype=float)
    errors = np.asarray(errors, dtype=float)
    order = np.argsort(h)
    h_s, e_s = h[order], errors[order]

    slope, intercept = np.polyfit(np.log(h_s), np.log(e_s), 1)
    fit_e = np.exp(intercept) * h_s**slope

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h_s,
            y=e_s,
            mode="markers+lines",
            name="observed error",
            meta="series-1",
            line=dict(color=SERIES[0], width=2),
            marker=dict(size=9, color=SERIES[0], line=dict(width=2, color="#fcfcfb")),
            hovertemplate="h=%{x:.4g}<br>error=%{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h_s,
            y=fit_e,
            mode="lines",
            name=f"fit (slope {slope:.2f})",
            meta="series-2",
            line=dict(color=SERIES[1], width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.06,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        text=f"observed convergence rate ≈ {slope:.2f}",
        font=dict(size=13, color="#52514e"),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(text="Convergence", x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(
            title=dict(text="h (mesh size)", font=dict(color="#52514e")),
            type="log",
        ),
        yaxis=dict(
            title=dict(text="L2 error", font=dict(color="#52514e")),
            type="log",
        ),
        legend=dict(font=dict(color="#52514e")),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    return _finish(fig, height=440)


# ---------------------------------------------------------------------------
# 3a. convergence_loglog_compare_html
# ---------------------------------------------------------------------------

def convergence_loglog_compare_html(series, title="Convergence: uniform vs adaptive", x_label="DOFs"):
    """Log-log error-vs-DOFs plot comparing MULTIPLE refinement strategies
    (e.g. uniform vs adaptive), each with its own fitted-slope annotation --
    the two-series extension of ``convergence_loglog_html`` (kept separate
    so that function's single-series signature/callers are undisturbed).

    Args:
        series: dict of {label: (dofs, errors)} -- each value a pair of
            equal-length 1D sequences, plotted in insertion order and
            colored by the dataviz categorical palette (``SERIES``).
        title: card title.
        x_label: x-axis title (DOFs by default; also sensible as "h" for a
            fixed-degree resolution sweep).
    """
    fig = go.Figure()
    slopes = {}
    for i, (label, (x_vals, y_vals)) in enumerate(series.items()):
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        order = np.argsort(x_vals)
        x_s, y_s = x_vals[order], y_vals[order]

        slope, intercept = np.polyfit(np.log(x_s), np.log(y_s), 1)
        slopes[label] = float(slope)
        fit_y = np.exp(intercept) * x_s**slope

        color = SERIES[(2 * i) % len(SERIES)]
        fig.add_trace(
            go.Scatter(
                x=x_s, y=y_s, mode="markers+lines", name=f"{label} (observed)",
                meta=f"series-{2 * i + 1}",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color, line=dict(width=2, color="#fcfcfb")),
                hovertemplate=f"{label}<br>{x_label}=%{{x:.4g}}<br>error=%{{y:.4g}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_s, y=fit_y, mode="lines", name=f"{label} fit (slope {slope:.3f})",
                meta=f"series-{2 * i + 2}",
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    annotation_lines = "<br>".join(f"{label}: slope ≈ {s:.3f}" for label, s in slopes.items())
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.06, xanchor="right", yanchor="bottom",
        showarrow=False, text=annotation_lines,
        font=dict(size=13, color="#52514e"), bgcolor="rgba(0,0,0,0)", align="right",
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text=x_label, font=dict(color="#52514e")), type="log"),
        yaxis=dict(title=dict(text="energy-norm error", font=dict(color="#52514e")), type="log"),
        legend=dict(font=dict(color="#52514e")),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    return _finish(fig, height=460)


# ---------------------------------------------------------------------------
# 3a-2. convergence_loglog_multi_order_html
# ---------------------------------------------------------------------------

def _reference_slope_triangle(fig, h_series, err_series, rate, color, meta):
    """Add a small log-log reference-slope triangle (dotted) near a series'
    coarsest two points, showing the THEORETICAL optimal rate for visual
    comparison against the series' own observed/fitted line -- distinct
    from the fitted dashed line (which is fit TO the data and so trivially
    tracks whatever the data does); this triangle is anchored to a fixed
    `rate` independent of the data, e.g. the textbook optimal L2 rate
    degree+1.
    """
    if len(h_series) < 2:
        return
    x0 = float(h_series[-2])
    x1 = float(h_series[-1])
    # Lift the triangle above the series' own curve so it doesn't overlap
    # the observed markers/line.
    y_at_x0 = float(err_series[np.argmin(np.abs(np.asarray(h_series) - x0))])
    lift = 2.4
    y0 = y_at_x0 * lift
    y1 = y0 * (x1 / x0) ** rate
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0],
            y=[y0, y0, y1, y0],
            mode="lines",
            meta=meta,
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.6,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=x1 * 1.12,
        y=(y0 * y1) ** 0.5,
        xref="x",
        yref="y",
        text=f"theory {rate:g}",
        showarrow=False,
        font=dict(size=10, color=color),
        xanchor="left",
    )


def convergence_loglog_multi_order_html(
    series, reference_rates, title="Convergence", x_label="h (mesh size)", y_label="L2 error"
):
    """Log-log error-vs-h plot comparing MULTIPLE element orders (e.g. P1,
    P2, P3 Lagrange), each with its own fitted-slope annotation AND a
    dotted reference triangle for its theoretical optimal rate -- the
    high-order-accuracy-verification extension of
    ``convergence_loglog_compare_html`` (kept separate so that function's
    existing signature/callers -- and its "energy-norm error" y-axis
    framing -- are undisturbed; this one is genuinely L2-error-flavored and
    needs the reference-triangle overlay compare_html doesn't have).

    Args:
        series: dict of {label: (h, errors)} -- each value a pair of
            equal-length 1D sequences, plotted in insertion order and
            colored by the dataviz categorical palette (``SERIES``). The
            fitted slope shown per series is a real least-squares fit
            (``np.polyfit``) over ALL points passed for that series.
        reference_rates: dict of {label: theoretical_rate} -- same keys as
            `series`; draws one dotted reference triangle per series at its
            theoretical optimal rate (e.g. {"P1": 2, "P2": 3, "P3": 4}).
        title: card title.
        x_label: x-axis title.
        y_label: y-axis title (e.g. "L2 error").
    """
    fig = go.Figure()
    slopes = {}
    for i, (label, (h_vals, err_vals)) in enumerate(series.items()):
        h_vals = np.asarray(h_vals, dtype=float)
        err_vals = np.asarray(err_vals, dtype=float)
        order = np.argsort(h_vals)
        h_s, e_s = h_vals[order], err_vals[order]

        slope, intercept = np.polyfit(np.log(h_s), np.log(e_s), 1)
        slopes[label] = float(slope)
        fit_e = np.exp(intercept) * h_s**slope

        color = SERIES[i % len(SERIES)]
        fig.add_trace(
            go.Scatter(
                x=h_s, y=e_s, mode="markers+lines", name=f"{label} (observed)",
                meta=f"series-{i + 1}",
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color, line=dict(width=2, color="#fcfcfb")),
                hovertemplate=f"{label}<br>h=%{{x:.4g}}<br>error=%{{y:.4g}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=h_s, y=fit_e, mode="lines", name=f"{label} fit (rate {slope:.2f})",
                meta=f"series-{i + 1}",
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )
        if label in reference_rates:
            _reference_slope_triangle(
                fig, h_s, e_s, reference_rates[label], color, meta=f"series-{i + 1}"
            )

    annotation_lines = "<br>".join(
        f"{label}: rate ≈ {s:.2f} (theory {reference_rates.get(label, '?')})"
        for label, s in slopes.items()
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.02, xanchor="left", yanchor="bottom",
        showarrow=False, text=annotation_lines,
        font=dict(size=13, color="#52514e"), bgcolor="rgba(0,0,0,0)", align="left",
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text=x_label, font=dict(color="#52514e")), type="log"),
        yaxis=dict(title=dict(text=y_label, font=dict(color="#52514e")), type="log"),
        legend=dict(font=dict(color="#52514e")),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    return _finish(fig, height=480)


# ---------------------------------------------------------------------------
# 3b. coefficient_timeseries_html
# ---------------------------------------------------------------------------

def coefficient_timeseries_html(times, series, title, y_label="value"):
    """Two-line (or N-line) time-series chart -- e.g. drag/lift coefficient
    vs simulated time, showing an oscillation (periodic vortex shedding)
    directly as a wiggling line, unlike a single converged-scalar readout.

    Args:
        times: (T,) sequence of simulated-time values (x axis).
        series: dict of {label: (T,) values} -- plotted in insertion order,
            colored by the dataviz categorical palette (``SERIES``).
        title: card title.
        y_label: y-axis title (e.g. "coefficient").
    """
    times = np.asarray(times, dtype=float)

    fig = go.Figure()
    for i, (label, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        color = SERIES[i % len(SERIES)]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=label,
                meta=f"series-{i + 1}",
                line=dict(color=color, width=2),
                hovertemplate=f"t=%{{x:.4g}}<br>{label}=%{{y:.4g}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text="t (simulated time)", font=dict(color="#52514e"))),
        yaxis=dict(title=dict(text=y_label, font=dict(color="#52514e"))),
        legend=dict(font=dict(color="#52514e")),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    return _finish(fig, height=420)


# ---------------------------------------------------------------------------
# 3b-2. scalar_sweep_html
# ---------------------------------------------------------------------------

def scalar_sweep_html(x, y, title, x_label="parameter", y_label="value", annotation=None):
    """Bar + line chart of a single derived scalar (e.g. net flow rate Q)
    against a swept parameter (e.g. occlusion amplitude) -- for a small
    (often 3-5 point) parameter sweep where ``convergence_loglog_html``'s
    log-log-with-fitted-slope framing doesn't apply (the sweep may include
    zero/negative-adjacent values, or the relationship isn't expected to be
    a power law).

    Args:
        x: (N,) swept parameter values.
        y: (N,) the resulting derived scalar at each x.
        title: card title.
        x_label, y_label: axis titles.
        annotation: optional preformatted string (e.g. "Q increases
            monotonically with amplitude") placed in the top-left corner.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_s, y=y_s, marker=dict(color=SERIES[0], line=dict(width=0)),
            opacity=0.55, hoverinfo="skip", showlegend=False, name="bar",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_s, y=y_s, mode="markers+lines", meta="series-1",
            line=dict(color=SERIES[0], width=2),
            marker=dict(size=10, color=SERIES[0], line=dict(width=2, color="#fcfcfb")),
            hovertemplate=f"{x_label}=%{{x:.4g}}<br>{y_label}=%{{y:.4g}}<extra></extra>",
            showlegend=False, name="value",
        )
    )
    if annotation:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.96, xanchor="left", yanchor="top",
            showarrow=False, text=annotation,
            font=dict(size=13, color="#52514e"), bgcolor="rgba(0,0,0,0)", align="left",
        )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text=x_label, font=dict(color="#52514e")), type="category" if len(x_s) <= 8 else "linear"),
        yaxis=dict(title=dict(text=y_label, font=dict(color="#52514e"))),
        margin=dict(l=70, r=30, t=50, b=50),
        bargap=0.5,
    )
    return _finish(fig, height=380)


# ---------------------------------------------------------------------------
# 3c. profile_with_fit_html
# ---------------------------------------------------------------------------

def profile_with_fit_html(
    x, measured, fit, title, x_label="x", y_label="value",
    measured_label="measured", fit_label="analytic fit", log_y=False, annotation=None,
):
    """1D profile chart: a measured curve (e.g. a steady-state morphogen
    gradient c(x), averaged/collapsed along any symmetric axis) overlaid
    with an analytic/theoretical curve (e.g. c0*exp(-x/lambda)) -- the
    single-line analogue of ``convergence_loglog_html``'s
    observed-vs-fit pairing, for profiles indexed by a spatial coordinate
    rather than mesh size.

    Args:
        x: (N,) sequence of x-coordinates (shared by both curves).
        measured: (N,) measured/simulated values.
        fit: (N,) analytic/theoretical values at the same `x`.
        title: card title.
        x_label, y_label: axis titles.
        measured_label, fit_label: legend labels for the two series.
        log_y: if True, use a log y-axis -- an exponential decay renders as
            a straight line, making deviations from pure exponential decay
            directly visible.
        annotation: optional preformatted string (e.g. "fitted lambda=0.32
            vs analytic 0.316") placed in the bottom-right corner.
    """
    x = np.asarray(x, dtype=float)
    measured = np.asarray(measured, dtype=float)
    fit = np.asarray(fit, dtype=float)
    order = np.argsort(x)
    x_s, m_s, f_s = x[order], measured[order], fit[order]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_s, y=m_s, mode="markers+lines", name=measured_label,
            meta="series-1",
            line=dict(color=SERIES[0], width=2),
            marker=dict(size=6, color=SERIES[0], line=dict(width=1, color="#fcfcfb")),
            hovertemplate=f"{x_label}=%{{x:.4g}}<br>{measured_label}=%{{y:.4g}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_s, y=f_s, mode="lines", name=fit_label,
            meta="series-2",
            line=dict(color=SERIES[1], width=2, dash="dash"),
            hovertemplate=f"{x_label}=%{{x:.4g}}<br>{fit_label}=%{{y:.4g}}<extra></extra>",
        )
    )
    if annotation:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.94, xanchor="right", yanchor="top",
            showarrow=False, text=annotation,
            font=dict(size=13, color="#52514e"), bgcolor="rgba(0,0,0,0)", align="right",
        )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text=x_label, font=dict(color="#52514e"))),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color="#52514e")),
            type="log" if log_y else "linear",
        ),
        legend=dict(font=dict(color="#52514e")),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    return _finish(fig, height=440)


# ---------------------------------------------------------------------------
# 3c-2. profile_compare_html
# ---------------------------------------------------------------------------

def profile_compare_html(series, title, x_label="x", y_label="value", log_y=False):
    """Multi-regime extension of ``profile_with_fit_html``: overlays SEVERAL
    (measured, analytic-fit) curve pairs -- e.g. a morphogen gradient at
    several decay lengths lambda -- each pair sharing one categorical color,
    solid for measured / dashed for the analytic curve (mirrors
    ``convergence_loglog_compare_html``'s observed/fit pairing per series).

    Args:
        series: dict of {label: (x, measured, fit)} -- each value a triple
            of equal-length 1D sequences, plotted in insertion order and
            colored by the dataviz categorical palette (``SERIES``).
        title: card title.
        x_label, y_label: axis titles.
        log_y: if True, use a log y-axis.
    """
    fig = go.Figure()
    for i, (label, (x_vals, measured, fit)) in enumerate(series.items()):
        x_vals = np.asarray(x_vals, dtype=float)
        measured = np.asarray(measured, dtype=float)
        fit = np.asarray(fit, dtype=float)
        order = np.argsort(x_vals)
        x_s, m_s, f_s = x_vals[order], measured[order], fit[order]

        color = SERIES[i % len(SERIES)]
        fig.add_trace(
            go.Scatter(
                x=x_s, y=m_s, mode="markers+lines", name=f"{label} (measured)",
                meta=f"series-{i + 1}",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color, line=dict(width=1, color="#fcfcfb")),
                hovertemplate=f"{label}<br>{x_label}=%{{x:.4g}}<br>{y_label}=%{{y:.4g}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_s, y=f_s, mode="lines", name=f"{label} (analytic)",
                meta=f"series-{i + 1}",
                line=dict(color=color, width=1.5, dash="dash"),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text=x_label, font=dict(color="#52514e"))),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color="#52514e")),
            type="log" if log_y else "linear",
        ),
        legend=dict(font=dict(color="#52514e")),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    return _finish(fig, height=460)


# ---------------------------------------------------------------------------
# 3d. french_flag_regions_html
# ---------------------------------------------------------------------------

def french_flag_regions_html(coords, values, thresholds, region_labels, title, boundary_x=None):
    """Positional-information readout: partitions a scalar field into
    ``len(thresholds) + 1`` fate regions via `thresholds` (descending),
    rendered as a discrete filled heatmap over the domain -- a literal nod
    to Wolpert's original "French flag" terminology (blue/white/red bands),
    used ONLY for this readout; every other function in this module follows
    the dataviz sequential/categorical palette.

    Args:
        coords: (N, 2) mesh dof coordinates.
        values: (N,) nodal field values (e.g. steady-state morphogen
            concentration).
        thresholds: sequence of threshold values, HIGH to LOW (e.g.
            (theta_high, theta_low)) -- defines 3 regions for 2 thresholds.
        region_labels: sequence of region names, same order as the regions
            they define (highest-value region first), length
            ``len(thresholds) + 1``.
        title: card title.
        boundary_x: optional sequence of x-positions to mark with dashed
            vertical reference lines (e.g. the ANALYTIC theta-crossing
            positions, for visual comparison against the rendered region
            edges).
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    thresholds_desc = sorted(thresholds, reverse=True)
    n_regions = len(thresholds_desc) + 1
    if len(region_labels) != n_regions:
        raise ValueError(f"expected {n_regions} region_labels, got {len(region_labels)}")

    xi, yi, zi = _interpolate_to_grid(coords, values, n=140)
    region = np.zeros_like(zi)
    for t in thresholds_desc:
        region += (zi < t).astype(float)  # 0 = highest-value region, ..., n_regions-1 = lowest

    default_colors = ["#1c5cab", "#f4f2e9", "#c0392b"]  # Wolpert's blue / white / red
    colors = (default_colors + default_colors)[:n_regions] if n_regions <= 3 else [
        SERIES[i % len(SERIES)] for i in range(n_regions)
    ]
    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i / n_regions, c])
        colorscale.append([(i + 1) / n_regions, c])

    fig = go.Figure(
        data=go.Heatmap(
            x=xi, y=yi, z=region, zmin=0, zmax=n_regions,
            colorscale=colorscale, showscale=False,
            connectgaps=False,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>region=%{z}<extra></extra>",
        )
    )
    for x0 in (boundary_x or []):
        fig.add_shape(
            type="line", x0=x0, x1=x0, y0=0, y1=1, yref="paper",
            line=dict(color="#0b0b0b", width=1.5, dash="dot"), opacity=0.6,
        )
    for i, label in enumerate(region_labels):
        mask = region == i
        x_center = float(xi[mask.any(axis=0)].mean()) if mask.any() else float(xi[len(xi) // (2 * n_regions)])
        fig.add_annotation(
            x=x_center, y=1.04, xref="x", yref="paper", showarrow=False,
            text=label, font=dict(size=13, color="#0b0b0b"),
        )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        xaxis=dict(title=dict(text="x", font=dict(color="#52514e")), scaleanchor="y", constrain="domain"),
        yaxis=dict(title=dict(text="y", font=dict(color="#52514e"))),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return _finish(fig, height=420)


# ---------------------------------------------------------------------------
# 4. quiver_streamlines_html
# ---------------------------------------------------------------------------

def quiver_streamlines_html(coords, u, v, speed, title):
    """Velocity field as arrows (direction) colored at the tip by speed (magnitude)."""
    coords = np.asarray(coords, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.asarray(speed, dtype=float)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    diag = float(np.hypot(xmax - xmin, ymax - ymin)) or 1.0
    max_speed = float(np.max(speed)) if speed.size and np.max(speed) > 0 else 1.0
    scale = 0.06 * diag / max_speed

    xs, ys = [], []
    for (x0, y0), du, dv in zip(coords, u, v):
        xs += [x0, x0 + du * scale, None]
        ys += [y0, y0 + dv * scale, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            meta="series-1",
            line=dict(width=2, color=SERIES[0]),
            opacity=0.5,
            hoverinfo="skip",
            showlegend=False,
            name="direction",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0] + u * scale,
            y=coords[:, 1] + v * scale,
            mode="markers",
            marker=dict(size=9, coloraxis="coloraxis", color=speed, line=dict(width=2, color="#fcfcfb")),
            hovertemplate="speed=%{marker.color:.4g}<extra></extra>",
            showlegend=False,
            name="speed",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0b0b0b")),
        coloraxis=dict(
            colorscale=_seq_colorscale(),
            colorbar=dict(title=dict(text="speed", font=dict(color="#52514e")), outlinewidth=0),
        ),
        xaxis=dict(title=dict(text="x", font=dict(color="#52514e")), scaleanchor="y", constrain="domain"),
        yaxis=dict(title=dict(text="y", font=dict(color="#52514e"))),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return _finish(fig, height=480)


# ---------------------------------------------------------------------------
# 5. mesh3d_html
# ---------------------------------------------------------------------------

_MESH3D_TEMPLATE = """
<div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
  <div style="font:600 15px system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--pbg-text);">__TITLE__</div>
  <div style="display:flex; align-items:center; gap:8px; font:12px system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--pbg-muted);">
    <span>__VMIN_LABEL__</span>
    <div style="width:120px; height:10px; border-radius:5px; background: linear-gradient(to right, __GRADIENT_CSS__);"></div>
    <span>__VMAX_LABEL__</span>
  </div>
</div>
<div id="__DIV_ID__" style="width:100%; height:__HEIGHT__px; border-radius:8px; overflow:hidden; background: var(--pbg-page);"></div>
__SLIDER_HTML__
<script src="__THREE_CDN__"></script>
<script src="__THREE_ORBIT_CDN__"></script>
<script>
(function () {
  var container = document.getElementById('__DIV_ID__');
  if (!container || typeof THREE === 'undefined') { return; }

  var positions = __POSITIONS_JSON__;
  var indices = __INDICES_JSON__;
  var framesValues = __VALUES_JSON__;
  var times = __TIMES_JSON__;
  var vmin = __VMIN__;
  var vmax = __VMAX__;
  var seqStops = __SEQ_STOPS_JSON__;

  function hexToRgb(hex) {
    var v = parseInt(hex.slice(1), 16);
    return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
  }
  var stopsRgb = seqStops.map(hexToRgb);

  function valueToColor(val) {
    var t = vmax > vmin ? (val - vmin) / (vmax - vmin) : 0.5;
    t = Math.max(0, Math.min(1, t));
    var n = stopsRgb.length - 1;
    var pos = t * n;
    var i0 = Math.floor(pos);
    var i1 = Math.min(i0 + 1, n);
    var frac = pos - i0;
    var c0 = stopsRgb[i0], c1 = stopsRgb[i1];
    return [
      c0[0] + (c1[0] - c0[0]) * frac,
      c0[1] + (c1[1] - c0[1]) * frac,
      c0[2] + (c1[2] - c0[2]) * frac
    ];
  }

  function buildColorArray(values) {
    var arr = new Float32Array(values.length * 3);
    for (var i = 0; i < values.length; i++) {
      var c = valueToColor(values[i]);
      arr[i * 3] = c[0]; arr[i * 3 + 1] = c[1]; arr[i * 3 + 2] = c[2];
    }
    return arr;
  }

  var posArray = new Float32Array(positions);
  var idxArray = new Uint32Array(indices);

  var geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
  geometry.setIndex(new THREE.BufferAttribute(idxArray, 1));
  geometry.setAttribute('color', new THREE.BufferAttribute(buildColorArray(framesValues[0]), 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  var scene = new THREE.Scene();
  var width = container.clientWidth || 600;
  var height = __HEIGHT__;
  var camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 10000);

  var bs = geometry.boundingSphere;
  var center = bs.center;
  var radius = bs.radius || 1;
  camera.position.set(center.x + radius * 1.6, center.y + radius * 1.2, center.z + radius * 1.6);
  camera.lookAt(center);

  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  var ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);
  var dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight.position.set(center.x + radius * 2, center.y + radius * 3, center.z + radius * 2);
  scene.add(dirLight);

  var material = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 6 });
  var mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  var wireMaterial = new THREE.MeshBasicMaterial({ color: 0x898781, wireframe: true, transparent: true, opacity: 0.15 });
  var wireMesh = new THREE.Mesh(geometry, wireMaterial);
  scene.add(wireMesh);

  var controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  function render() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () {
      var w = container.clientWidth, h = container.clientHeight;
      if (!w || !h) { return; }
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container);
  }

  var slider = document.getElementById('__DIV_ID__-slider');
  if (slider && times) {
    slider.addEventListener('input', function () {
      var idx = parseInt(slider.value, 10);
      geometry.setAttribute('color', new THREE.BufferAttribute(buildColorArray(framesValues[idx]), 3));
      var label = document.getElementById('__DIV_ID__-slider-label');
      if (label) { label.textContent = 't = ' + times[idx]; }
    });
  }
  var playBtn = document.getElementById('__DIV_ID__-play');
  if (playBtn && times) {
    var playing = false, timer = null;
    playBtn.addEventListener('click', function () {
      playing = !playing;
      playBtn.textContent = playing ? '\\u23F8' : '\\u25B6';
      if (playing) {
        timer = setInterval(function () {
          var idx = (parseInt(slider.value, 10) + 1) % times.length;
          slider.value = idx;
          slider.dispatchEvent(new Event('input'));
        }, 600);
      } else if (timer) {
        clearInterval(timer);
      }
    });
  }
})();
</script>
"""

_MESH3D_SLIDER = """
<div style="display:flex; align-items:center; gap:10px; margin-top:10px;">
  <button id="__DIV_ID__-play" style="border:1px solid var(--pbg-border); background:var(--pbg-page); color:var(--pbg-text); border-radius:6px; padding:4px 10px; cursor:pointer;">▶</button>
  <input id="__DIV_ID__-slider" type="range" min="0" max="__MAXIDX__" value="0" step="1" style="flex:1;">
  <span id="__DIV_ID__-slider-label" style="font:12px system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--pbg-muted); min-width:90px; text-align:right;">t = __T0__</span>
</div>
"""


def mesh3d_html(coords3, cells, values, times=None, title="3D field"):
    """Three.js triangulated 3D mesh viewer: orbit controls, sequential
    colormap, optional time slider, low-opacity wireframe overlay.

    Args:
        coords3: (N, 3) vertex coordinates.
        cells: (M, 3) triangle vertex-index connectivity.
        values: (N,) nodal values (static), or a sequence of (N,) arrays
            (one per entry in `times`) when `times` is given.
        times: optional sequence of time labels; when given, `values` must
            be a sequence of per-time nodal arrays of the same length.
        title: card title.
    """
    coords3 = np.asarray(coords3, dtype=float)
    cells = np.asarray(cells, dtype=int)

    if times is None:
        values_frames = [np.asarray(values, dtype=float)]
        times_list = None
    else:
        values_frames = [np.asarray(v, dtype=float) for v in values]
        times_list = [float(t) for t in times]

    vmin = float(min(float(np.nanmin(v)) for v in values_frames))
    vmax = float(max(float(np.nanmax(v)) for v in values_frames))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    div_id = _new_id("pbg-mesh3d")
    gradient_css = ", ".join(_SEQ_STOPS)

    if times_list is None:
        slider_html = ""
    else:
        slider_html = (
            _MESH3D_SLIDER.replace("__DIV_ID__", div_id)
            .replace("__MAXIDX__", str(len(times_list) - 1))
            .replace("__T0__", f"{times_list[0]:.4g}")
        )

    html = _MESH3D_TEMPLATE
    html = html.replace("__TITLE__", str(title))
    html = html.replace("__VMIN_LABEL__", f"{vmin:.4g}")
    html = html.replace("__VMAX_LABEL__", f"{vmax:.4g}")
    html = html.replace("__GRADIENT_CSS__", gradient_css)
    html = html.replace("__DIV_ID__", div_id)
    html = html.replace("__HEIGHT__", "480")
    html = html.replace("__SLIDER_HTML__", slider_html)
    html = html.replace("__THREE_CDN__", THREE_CDN)
    html = html.replace("__THREE_ORBIT_CDN__", THREE_ORBIT_CDN)
    html = html.replace("__POSITIONS_JSON__", json.dumps(coords3.flatten().tolist()))
    html = html.replace("__INDICES_JSON__", json.dumps(cells.flatten().tolist()))
    html = html.replace("__VALUES_JSON__", json.dumps([v.tolist() for v in values_frames]))
    html = html.replace("__TIMES_JSON__", json.dumps(times_list))
    html = html.replace("__VMIN__", repr(vmin))
    html = html.replace("__VMAX__", repr(vmax))
    html = html.replace("__SEQ_STOPS_JSON__", json.dumps(_SEQ_STOPS))

    return _card_open() + html + _card_close()
