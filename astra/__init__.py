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
from importlib import metadata

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"

# astropy import style
def __getattr__(attr):
    if attr in __all__:
        from importlib import import_module

        return import_module('astra.' + attr)

    raise AttributeError(f"module 'astra' has no attribute {attr!r}")


# redefine dir
def __dir__():
    return sorted(set(globals()).union(__all__))
