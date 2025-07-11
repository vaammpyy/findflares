from pipeline_utils import *

global data_dir

candidates_file="/home/vampy/acads/projects/Superflares/Data/lists/TIC_G_Type_test.txt"
telescope="TESS"
data_dir="/home/vampy/acads/projects/Superflares/Data"
output_dir=data_dir
CPU_CORES=2

run_pipeline_py(candidates_file, telescope, data_dir,output_dir,CPU_CORES, redo=False, injrec=0)