from TESSLC_class import *
from imports import *

# # tic=TESSLC(118327550)
# tic=TESSLC(98796344)
# # tic.download_lc(sector=29, segment=True, clean=True)
# tic.download_lc(sector=31, segment=True, clean=True)
# tic.detrend_2()
# tic.findflares()
# tic.pickleObj()

def pipeline(tic, sector):
    lc=TESSLC(tic)
    lc.download_lc(sector=sector, segment=True, clean=True)
    lc.detrend_2()
    lc.findflares()
    lc.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True)
    lc.plot(mode="model_overlay", save_fig=True)
    lc.pickleObj()

target_list_file="/home/vampy/acads/projects/Stellar Flares/Data/targets/rotating.txt"

# making a list of all tics
with open(target_list_file, "r") as file:
    tics=[int(line.strip(" ")) for line in file]

for tic in tics:
    print(f"TIC {tic}")
    search_result = search_lightcurve(tic, ret_list=True)
    if search_result:
        for result in search_result:
            TIC, sector = result
            print(f"Sector {sector}")
            print("Pipeline Started.")
            pipeline(TIC, sector)
            print("Pipeline Completed.")
    else:
        print("No data found!")
