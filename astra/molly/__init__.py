from .analysis import do_xcor, do_optsub, do_ews
from astra.utils._helpers import deprecated_import

msg = (f"{__name__} is deprecated, and will be removed in a future version. "
	"Equivalent and/or improved methods are available in astra.correlate and astra.fitting.")

deprecated_import(msg)