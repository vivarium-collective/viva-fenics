import numpy as np
from pbg_fenics import fem


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
