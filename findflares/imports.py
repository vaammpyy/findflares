import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import lightkurve as lk
from astropy.time import Time
import pymc as pm
import pymc_ext as pmx
import pytensor.tensor as tt 
from celerite2.pymc import terms, GaussianProcess
from requests.exceptions import HTTPError, ConnectionError
import warnings
import copy

from .aflare import aflare, aflare1
from .FINDflare_dport import FINDflare
from .download import *
from .lc_utils import *
from .detrending_utils import *
from .flares_utils import *
from .injection_recovery import *

warnings.filterwarnings("ignore", category=FutureWarning)