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

parser.add_argument('-e','--calc_energy', type=str2bool, default=True, help='True to calculate flare energy.')

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

parser.add_argument('-p', '--period',
                    type=float,
                    default=0,
                    # action='store_true',
                    help='Rotation period of the star.')

parser.add_argument('-f', '--distance',
                    type=float,
                    default=0,
                    # action='store_true',
                    help='Distance of the star in pc.')

parser.add_argument('-l', '--lc-dir',
                    type=str,
                    default=None,
                    # action='store_true',
                    help='Path to downloaded lightcurve.')
# Step 3: Parse the arguments
args = parser.parse_args()

rerun=args.rerun
calc_energy=args.calc_energy
DATA_dir= args.datadir
injrec=args.injrec
input_sector=args.sector
input_cadence=args.cadence
input_period = args.period
input_distance = args.distance
input_lc_dir = args.lc_dir

tess_pipeline(args.tic, DATA_dir, rerun, injrec, input_cadence, input_sector, calc_energy=calc_energy, period=input_period, distance=input_distance, lc_dir=input_lc_dir)