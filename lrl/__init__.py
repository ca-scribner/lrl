from .environments import *
from .data_stores import *
from .utils import *
from .solvers import *

# Taken from Requests:
# Set default logging handler to avoid "No handler found" warnings.
# Recommendation from python docs and using Requests package as a template
import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())
