import os
import glob.glob as glob

import .lc_class

def make_flare_list(stars_dir, flare_file):
    stars_folders=sorted(glob(stars_dir))
    print (stars_folders)