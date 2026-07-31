"""Smoke tests for pbg_fenics.viz -- shape/content checks on the returned
HTML fragments, not visual assertions. Real rendering is verified by hand
(see task-6-report.md) since headless browser rendering isn't in scope here.
"""

import numpy as np
import pytest

from pbg_fenics import viz


def test_field_heatmap_html():
    html = viz.field_heatmap_html(np.random.rand(20, 2), np.random.rand(20), "t")
    assert "plotly" in html.lower() and len(html) > 500


def test_field_heatmap_html_has_title_and_colorbar():
    html = viz.field_heatmap_html(np.random.rand(30, 2), np.random.rand(30), "Temperature field")
    assert "Temperature field" in html
    assert "coloraxis" in html


def test_convergence_loglog_html():
    html = viz.convergence_loglog_html([1 / 8, 1 / 16, 1 / 32], [1e-2, 2.5e-3, 6e-4])
    assert "plotly" in html.lower()


def test_convergence_loglog_html_reports_fitted_slope():
    # h halves each step, error drops by ~4x => second-order convergence, slope ~2.
    h = [1 / 8, 1 / 16, 1 / 32]
    errors = [1e-2, 2.5e-3, 6.25e-4]
    html = viz.convergence_loglog_html(h, errors)
    assert "2.0" in html or "1.9" in html or "2.00" in html
    assert "log" in html.lower()


def test_field_animation_html():
    coords = np.random.rand(25, 2)
    frames = [np.random.rand(25) for _ in range(4)]
    times = [0.0, 0.1, 0.2, 0.3]
    html = viz.field_animation_html(coords, frames, times, "Diffusing field")
    assert "plotly" in html.lower() and len(html) > 500
    assert "Diffusing field" in html
    assert "Plotly.animate" in html or "frames" in html.lower()


def test_quiver_streamlines_html():
    coords = np.random.rand(15, 2)
    u = np.random.randn(15)
    v = np.random.randn(15)
    speed = np.sqrt(u**2 + v**2)
    html = viz.quiver_streamlines_html(coords, u, v, speed, "Velocity field")
    assert "plotly" in html.lower() and len(html) > 500
    assert "Velocity field" in html


def test_mesh3d_html_static():
    coords3 = np.random.rand(10, 3)
    cells = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    values = np.random.rand(10)
    html = viz.mesh3d_html(coords3, cells, values)
    assert "three" in html.lower() and len(html) > 500
    assert "OrbitControls" in html
    assert "wireframe" in html.lower()


def test_mesh3d_html_with_times_includes_slider():
    coords3 = np.random.rand(10, 3)
    cells = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    values = [np.random.rand(10) for _ in range(3)]
    times = [0.0, 1.0, 2.0]
    html = viz.mesh3d_html(coords3, cells, values, times=times)
    assert "three" in html.lower()
    assert "range" in html.lower()  # <input type="range"> slider present


@pytest.mark.parametrize("fn_name", [
    "field_heatmap_html",
    "field_animation_html",
    "convergence_loglog_html",
    "quiver_streamlines_html",
    "mesh3d_html",
])
def test_all_viz_functions_exist(fn_name):
    assert hasattr(viz, fn_name)
    assert callable(getattr(viz, fn_name))
