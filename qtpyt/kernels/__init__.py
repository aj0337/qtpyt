"""Low-level kernels for the Green's-function solvers, CPU or GPU depending on ``qtpyt.xp``."""
from qtpyt import xp

if xp.__name__ == "numpy":
    from .linalg_cpu import solve, lrsolve, tsolve
    from .hs_cpu import geamC, geamT, get_lambda

else:
    from .linalg_gpu import solve, lrsolve, tsolve
    from .hs_gpu import geamC, geamT, get_lambda
