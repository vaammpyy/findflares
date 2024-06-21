from TESSLC_class import *
from imports import *

# tic=TESSLC(118327550)
tic=TESSLC(98796344)
# tic.download_lc(sector=29, segment=True, clean=True)
tic.download_lc(sector=31, segment=True, clean=True)
tic.detrend_2()
tic.findflares()
tic.pickleObj()
