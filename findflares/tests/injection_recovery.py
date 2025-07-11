from findflares.pipeline_utils import run_pipeline_py

candidates_file="./stars_list.txt"
telescope="TESS"
data_dir="./output"
output_dir=data_dir
CPU_CORES=2
INJREC=3
redo=True

run_pipeline_py(candidates_file, telescope, data_dir,output_dir,CPU_CORES, redo=redo, injrec=INJREC)