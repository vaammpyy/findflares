import pdb
import numpy as np
import pickle
import os
import lightkurve as lk
from astropy.time import Time
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt 
from celerite2.theano import terms, GaussianProcess
from aflare import aflare, aflare1