import argparse

from .lc_class import *
from .imports import *

# Step 1: Create the parser
parser = argparse.ArgumentParser(prog='FindFlare Pipeline',
                                 description='Finds flares in TESS lightcurve')

parser.add_argument('-t', "--tic",
                    type=int,
                    help='TIC of the target star.')

parser.add_argument('-r', "--rerun",
                    action='store_true',
                    help='Re-run pipeline for all existing stars.')

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
DATA_dir= data_dir
injrec=args.injrec
input_sector=args.sector
input_cadence=args.cadence

print(f"RE-RUN::{rerun}")
print(f"Inj-Rec::{injrec}")

def run_pipeline(tic, data_dir, redo=True, injrec=0, input_cadence=0, input_sector=0):
    """
    Runs pipeline for TESS Lightcurves.

    Runs pipeline for TESS Lightcurves, and stores output in data_dir.

    Parameters
    ----------
    tic : int
        TIC
    data_dir : str
        Directory to store the results.
    redo : bool, optional
        If True re-runs the pipeline for existing TICs.
    injrec : int, optional
        Runs Injection Recovery if value is 1, by default value is 0.
    input_cadence : int, optional
        Cadence of the observation for analysis, by default value is 0
        pipeline runs over both 120s and 20s data.
    input_sector : int, optional
        Sector of the observation for analysis, by default value is 0
        pipeline runs over all avialable sectors.
    """
    print("Pipeline Started.")
    print("###################")
    print(f"TIC {tic}")
    cad_list=[20, 120]
    print(f"Searching for observations with CADENCE: {cad_list}")
    try:
        if input_cadence and input_sector:
            search_result = [(tic, input_sector, input_cadence)]
        else:
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
                    try:
                        lc.download_lc(sector=sector, cadence=cad, clean=True)
                    except HTTPError:
                        print("Download failed. Donwloading next sector.")
                        continue
                    except ConnectionError:
                        print("ConnectionError, failed to connect to the server.")
                        continue
                    lc.detrend_3()
                    lc.findflares()
                    lc.flare_energy()
                    if injrec:
                        print("Injection recovery test started.")
                        irec=InjRec(lc)
                        for k in range(injrec):
                            irec.run_injection_recovery(run=k+1, plot=False)
                        print("Injection recovery test completed.")
                        plot_ir_results(irec, mode='rec_frac', save_fig=True)
                        plot_ir_results(irec, mode='erg_comp', save_fig=True)
                        plot_ir_results(irec, mode='fp', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_erg', save_fig=True)
                        irec.pickleObj()
                    else:
                        lc.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="model_overlay", save_fig=True)
                        lc.pickleObj()
                    print("****************")
        else:
            print("No data found!")
        print("Pipeline Completed.")
    except HTTPError:
        print("HTTPError, failed to fetch data.")
    except ConnectionError:
        print("ConnectionError, failed to connect to the server.")

run_pipeline(args.tic, data_dir, rerun, injrec, input_cadence, input_sector)