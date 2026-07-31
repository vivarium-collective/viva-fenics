"""Custom bigraph-schema types for viva-fenics.

`fem_field` represents nodal values of a dolfinx Function (e.g. a Poisson
solution) as a flat float64 array, for use as a process-bigraph port type.
"""

FENICS_TYPES = {
    "fem_field": {
        "_inherit": "array",
        "_data": "float64",
    },
}


def register_types(core):
    """Register viva-fenics bigraph-schema types into a process-bigraph core.

    Registered defensively (skipped if already present) so importing
    viva-fenics into another workspace does not clash.
    """
    try:
        existing = set(core.types())
    except Exception:
        existing = set()
    to_register = {k: v for k, v in FENICS_TYPES.items() if k not in existing}
    if to_register:
        core.register_types(to_register)
    return core
