import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

import astra.correlate as correlate
import astra.fitting as fitting
import astra.integrate as integrate
import astra.plotting as plotting
import astra.utils as utils
