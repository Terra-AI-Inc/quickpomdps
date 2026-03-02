import builtins
import os

import julia

_original_import = builtins.__import__


def _auto_install_julia_import(name, *args, **kwargs):
    """Import hook that auto-installs Julia packages on first use.

    When ``from julia.POMCPOW import ...`` fails because POMCPOW isn't
    installed, this runs ``Pkg.add("POMCPOW")`` and retries.
    """
    try:
        return _original_import(name, *args, **kwargs)
    except ImportError as original_err:
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "julia":
            pkg = parts[1]
            try:
                from julia import Pkg

                Pkg.add(pkg)
                return _original_import(name, *args, **kwargs)
            except Exception:
                raise original_err
        raise


builtins.__import__ = _auto_install_julia_import


def install_julia_dependencies():
    """Install Julia packages required for quickpomdps."""
    from julia import Pkg

    Pkg.add(["PyCall", "POMDPs", "POMDPTools"])


def require_julia_package(*packages):
    """Install Julia packages on demand. Prefer using normal imports instead —
    ``from julia.X import ...`` auto-installs missing packages."""
    from julia import Pkg

    Pkg.add(list(packages))


try:
    from julia import Main as _Main
except Exception as ex:
    print("Could not load Julia.")
    print("Caught the following exception:")
    print(ex)
    if isinstance(ex, julia.core.UnsupportedPythonError):
        print("Running julia.install()")
        julia.install()
    print("Running quickpomdps.install_julia_dependencies()")
    install_julia_dependencies()
    print("done!")
    from julia import Main as _Main

# Ensure base dependencies are available
try:
    _Main.eval("using POMDPs; using POMDPTools")
except Exception:
    install_julia_dependencies()
    _Main.eval("using POMDPs; using POMDPTools")

_script_dir = os.path.dirname(os.path.realpath(__file__))
_Main.include(os.path.join(_script_dir, "bridge.jl"))
_Main.eval("using .PyPOMDPsBridge")
_create_pypomdp = _Main.eval("PyPOMDPsBridge.create_pypomdp")
_create_pymdp = _Main.eval("PyPOMDPsBridge.create_pymdp")


class POMDP:
    """Python wrapper that creates a Julia POMDP via bridge.jl.

    Pass to Julia solvers via the ``.jl`` attribute.
    """

    def __init__(self, **kwargs):
        self._jl = _create_pypomdp(**kwargs)

    @property
    def jl(self):
        return self._jl


class MDP:
    """Python wrapper that creates a Julia MDP via bridge.jl.

    Pass to Julia solvers via the ``.jl`` attribute.
    """

    def __init__(self, **kwargs):
        self._jl = _create_pymdp(**kwargs)

    @property
    def jl(self):
        return self._jl
