"""
LIFE E2E Nulling Interferometer — Analytical Beam Propagation Model
====================================================================

End-to-end analytical wavefront propagation model for the Large
Interferometer For Exoplanets (LIFE) nulling combiner.

Modules
-------
material_properties : Wavelength-dependent optical constants (Au, CaF₂, ZnSe, ...)
fiber_modes         : Single-mode fiber optics (V-parameter, mode field radius)
m1_fiber_coupling   : Fiber coupling efficiency analysis
m2_throughput_chain : Surface-by-surface throughput waterfall
m3_null_error_propagation : Null depth error budget
m4_surface_sensitivity    : Surface WFE sensitivity ranking
monte_carlo               : Full end-to-end Monte Carlo integration

Author: Victor Huarcaya (University of Bern)
Paper:  Huarcaya (2026), A&A, in preparation
        "Analytical Throughput, Null Depth, and Surface Tolerance Budget
         for the LIFE Nulling Interferometer Combiner"
"""

__version__ = "1.0.0"
__author__ = "Victor Huarcaya"
__email__ = "victor.huarcaya@unibe.ch"
