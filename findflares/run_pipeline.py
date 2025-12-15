import argparse

from findflares.lc_class import *
from findflares.imports import *
from findflares.pipeline_utils import tess_pipeline

# Step 1: Create the parser
parser = argparse.ArgumentParser(prog='FindFlare Pipeline',
                                 description='Finds flares in TESS lightcurve')

parser.add_argument('-t', "--tic",
                    type=int,
                    help='TIC of the target star.')

parser.add_argument('-d', "--datadir",
                    type=str,
                    help="Path to store the output")

# parser.add_argument('-r', "--rerun",
#                     action='store_true',
#                     help='Re-run pipeline for all existing stars.')
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument('-r','--rerun', type=str2bool, default=False, help='True to rerun the pipeline for already processed data.')

parser.add_argument('-i', '--injrec',
                    type=int,
                    default=0,
                    # action='store_true',
                    help='Number of injection recovery test runs.')

parser.add_argument('-s', '--sector',
                    type=int,
                    default=0,
                    # action='store_true',
                    help='Observation sector for the data.')

parser.add_argument('-c', '--cadence',
                    type=int,
                    default=0,
                    # action='store_true',
                    help='Observation cadence for the data.')

# Step 3: Parse the arguments
args = parser.parse_args()

rerun=args.rerun
DATA_dir= args.datadir
injrec=args.injrec
input_sector=args.sector
input_cadence=args.cadence

tess_pipeline(args.tic, DATA_dir, rerun, injrec, input_cadence, input_sector)