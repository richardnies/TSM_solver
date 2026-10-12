"""Evaluation functions used by the examples (examples/<name>/<name>.py).

Each example is a script with its parameters at the top: it builds the geometry, the primary and the dispersion
relation with tsm_disp and calls these functions to follow roots, save the data and draw the PDF figures next to
itself. Installed with tsm_disp (a second top-level package; the figures need matplotlib, extra `plot`), so the
examples run from any directory and from copies outside the repository.

    plotting       figure style and PDF saving with a provenance line
    kx_scan        continuation of a root in k_x (fixed grid or adaptive step), ω(k_x) data and figure
    mode_structure eigenmode on the flux surface: data and figure
    root_search    all roots in a box (tsm_disp.contour_roots): summary and map of the search
"""
