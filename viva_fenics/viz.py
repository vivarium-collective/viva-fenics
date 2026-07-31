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

def field_animation_html(coords, frames, times, title):
    """Plotly animation of a scalar field over time, with a slider + play/pause."""
    frames = [np.asarray(f, dtype=float) for f in frames]
    times = list(times)

    xi = yi = None
    zis = []
    for f in frames:
        xi, yi, zi = _interpolate_to_grid(coords, f)
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
