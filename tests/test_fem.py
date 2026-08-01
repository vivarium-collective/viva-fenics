import ufl
import numpy as np
from viva_fenics import fem


def test_poisson_mms_converges():
    domain, V = fem.build_mesh("unit_square", 16, degree=2)
    uh = fem.solve_poisson(
        domain, V,
        source_fn=lambda x: -6.0 + 0*x[0],
        bc_fn=lambda x: 1 + x[0]**2 + 2*x[1]**2,
    )
    err = fem.l2_error(domain, V, uh, lambda x: 1 + x[0]**2 + 2*x[1]**2)
    assert err < 1e-10
    assert fem.node_coords(V).shape == (uh.size, 2)


def _smooth_trig_l2_error(resolution, degree):
    domain, V = fem.build_mesh("unit_square", resolution, degree=degree)
    uh = fem.solve_poisson(
        domain, V,
        source_fn=lambda x: 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
        bc_fn=lambda x: 0.0 * x[0],
    )
    return fem.l2_error_exact(
        domain, V, uh, lambda x: ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    )


def test_l2_error_exact_converges_at_optimal_rate_p1():
    e_coarse = _smooth_trig_l2_error(16, degree=1)
    e_fine = _smooth_trig_l2_error(32, degree=1)
    # h halves -> P1 (degree+1=2) optimal L2 error should drop ~4x.
    rate = np.log2(e_coarse / e_fine)
    assert e_fine < e_coarse
    assert abs(rate - 2.0) <= 0.25


def test_l2_error_exact_not_quadrature_limited_at_p3():
    # A stale/insufficient quadrature degree would flatten P3's error as
    # resolution increases (dominated by quadrature error instead of
    # discretization error) -- assert it keeps dropping sharply instead.
    e_coarse = _smooth_trig_l2_error(16, degree=3)
    e_fine = _smooth_trig_l2_error(32, degree=3)
    rate = np.log2(e_coarse / e_fine)
    assert e_fine < e_coarse
    assert abs(rate - 4.0) <= 0.25
