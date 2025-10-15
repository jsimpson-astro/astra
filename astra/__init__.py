__all__ = [
    '__version__',
    # '__bibtex__',
    # subpackages (lazy-loaded?)
    'correlate',
    'fitting',
    'integrate',
    'plotting',
    'utils'
]

# version
import importlib.metadata
__version__ = importlib.metadata.version(__package__ or __name__)
# clean up namespace
del importlib.metadata

# astropy import style
def __getattr__(attr):
    if attr in __all__:
        from importlib import import_module

        return import_module('astra.' + attr)

    raise AttributeError(f"module 'astra' has no attribute {attr!r}")


# redefine dir
def __dir__():
    return sorted(set(globals()).union(__all__))


from types import ModuleType as __module_type__

# clean up top-level namespace
# delete everything not in __all__
# or is a built-in attribute
# or that isn't a submodule of this package
for varname in dir():
    if not (
        (varname.startswith('__') and varname.endswith('__'))
        or varname in __all__
        or (
            varname[0] != '_'
            and isinstance(locals()[varname], __module_type__)
            and locals()[varname].__name__.startswith(__name__ + '.')
        )
    ):
        del locals()[varname]

del varname, __module_type__
