from .environments import *
from .data_stores import *
from .utils import *
from .solvers import *

# FUTURE: Am I handling logger correctly?  See refs:
#   https://stackoverflow.com/questions/27016870/how-should-logging-be-used-in-a-python-package
#   https://docs.python.org/3.7/howto/logging.html#configuring-logging-for-a-library
#   https://github.com/kennethreitz/requests/blob/master/requests/__init__.py

# Taken from Requests:
# Set default logging handler to avoid "No handler found" warnings.
# Recommendation from python docs and using Requests package as a template
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
