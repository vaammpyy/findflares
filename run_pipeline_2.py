from TESSLC_class import *
from imports import *
import argparse

# Step 1: Create the parser
parser = argparse.ArgumentParser(description='TIC')

# Step 2: Add an integer argument
parser.add_argument('tic', type=int, help='TIC of the target star.')

# Step 3: Parse the arguments
args = parser.parse_args()

global redo
global DATA_dir

redo=REDO
DATA_dir= data_dir

def pipeline(tic):
    print("Pipeline Started.")
    print("###################")
    print(f"TIC {tic}")
    cad_list=[20, 120]
    print(f"Searching for observations with CADENCE: {cad_list}")
    search_result = search_lightcurve(tic, cadence=cad_list, ret_list=True)
    if search_result:
        for result in search_result:
            TIC, sector, cad = result
            # checking if the pipeline has already run or not
            if os.path.isfile(f"{data_dir}/{TIC}/{sector}_{cad}.pkl") and redo==False:
                print(f"Sector {sector}, Cadence {cad}")
                print("Already exists.")
            else:
                print(f"Sector {sector}, Cadence {cad}")
                # pipeline(TIC, sector, cad)
                print("****************")
                lc=TESSLC(tic)
                # lc.download_lc(sector=sector, cadence=cadence, segment=True, clean=True)
                lc.download_lc(sector=sector, cadence=cad, clean=True)
                lc.detrend_3()
                lc.findflares()
                lc.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True)
                lc.plot(mode="model_overlay", save_fig=True)
                lc.pickleObj()
                print("****************")
    else:
        print("No data found!")
    print("Pipeline Completed.")

pipeline(args.tic)