from TESSLC_class import *
from imports import *

def pipeline(tic, sector, cadence):
    print("****************")
    lc=TESSLC(tic)
    # lc.download_lc(sector=sector, cadence=cadence, segment=True, clean=True)
    lc.download_lc(sector=sector, cadence=cadence, clean=True)
    lc.detrend_3()
    lc.findflares()
    lc.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True)
    lc.plot(mode="model_overlay", save_fig=True)
    lc.pickleObj()
    print("****************")

# making a list of all tics
with open(target_list_file, "r") as file:
    tics=[int(line.strip(" ")) for line in file]

# if True then processing will be done again for all the targets
REDO=REDO

print("Pipeline Started.")
for tic in tics:
    print("###################")
    print(f"TIC {tic}")
    cad_list=[20, 120]
    print(f"Searching for observations with CADENCE: {cad_list}")
    search_result = search_lightcurve(tic, cadence=cad_list, ret_list=True)
    if search_result:
        for result in search_result:
            TIC, sector, cad = result
            # checking if the pipeline has already run or not
            if os.path.isfile(f"{data_dir}/{TIC}/{sector}_{cad}.pkl") and REDO==False:
                print(f"Sector {sector}, Cadence {cad}")
                print("Already exists.")
            else:
                print(f"Sector {sector}, Cadence {cad}")
                pipeline(TIC, sector, cad)
    else:
        print("No data found!")
print("Pipeline Completed.")