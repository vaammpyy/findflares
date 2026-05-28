from mpi4py import MPI
import os
import sys
from pathlib import Path
import time
import random

os.environ["FLEXIBLAS"] = "openblas"
os.environ["FLEXIBLAS64"] = "openblas"

# # setting up the MPI environment
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # 2. Grab the live Slurm Job ID
# job_id = os.environ.get("SLURM_JOB_ID", "local")


# # 3. Map this specific task to its own isolated sandbox folder on scratch
# unique_task_dir = f"/home/krohan/scratch/pytensor_job_{job_id}/task_{rank}"
# os.makedirs(unique_task_dir, exist_ok=True)

# # 4. Bind PyTensor strictly to this sandbox path
# os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={unique_task_dir}"

# 1. Setting up the MPI environment correctly
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 2. Grab the live Slurm Job ID (None if running locally)
job_id = os.environ.get("SLURM_JOB_ID")

# 3. Determine the base directory depending on the environment
if job_id:
    # Cluster environment: High-speed scratch space
    base_dir = Path(f"/home/krohan/scratch/pytensor_job_{job_id}")
else:
    # Local Linux environment: Uses ~/.cache/pytensor_local
    home_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    base_dir = Path(home_cache) / "pytensor_local"

# 4. CRITICAL FIX: Use the actual MPI rank for isolation, NOT SLURM_PROCID
unique_task_dir = base_dir / f"task_{rank}"
unique_task_dir.mkdir(parents=True, exist_ok=True)

# 5. Bind PyTensor strictly to this sandbox path
os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={unique_task_dir}"
print(f"Rank {rank}: PyTensor compile directory set to: {unique_task_dir}")

# adding a jitter to desynchronise the processes.
jitter_time = random.uniform(0.1, 5.0)
time.sleep(jitter_time)

import pandas as pd
import argparse
import numpy as np
import traceback
import psutil
from astropy.io import fits
fits.Conf.use_memmap.set(False)

from findflares.pipeline_utils import tess_pipeline_mpi

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(prog='FindFlare Pipeline',
                                 description='Finds flares in TESS lightcurve')

parser.add_argument('-d', "--datadir",
                    type=str,
                    help="Path to store the output files.")

parser.add_argument('-l', "--logdir",
                    type=str,
                    help="Path to store the log files.")

parser.add_argument('-r','--rerun', type=str2bool, default=False, help='True to rerun the pipeline for already processed data.')

parser.add_argument('-i', '--injrec',
                    type=int,
                    default=0,
                    # action='store_true',
                    help='Number of injection recovery test runs.')

parser.add_argument('-t', "--target_path",
                    type=str,
                    help="Path to the target file.")

args = parser.parse_args()

DATA_DIR = args.datadir
LOG_DIR = args.logdir
rerun = args.rerun
injrec = args.injrec
TARGET_PATH = args.target_path

def get_memory_usage():
    """Returns the current RAM usage of the process in Megabytes."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)  # Convert to MB

def main():
    """
    Main function which spawns the process.
    """
    if rank == 0:
        job_id = os.environ.get('SLURM_JOB_ID', 'Local')
        # Format: YYYYMMDD_HHMM_JobID
        timestamp = time.strftime("%Y%m%d_%H%M")
        log_dir = LOG_DIR+f"/run_{timestamp}_Job_{job_id}"
        os.makedirs(log_dir, exist_ok=True)
        print(f"Master: Creating log directory: {log_dir}")
    else:
        log_dir = None

    # broadcast the log directory to each of the nodes
    log_dir = comm.bcast(log_dir, root=0)

    # Strategy for the Master Rank is to load the table and split it into chunks
    # and parse the relevant columns to the worker ranks.
    if rank == 0:
        master_log_path = os.path.join(log_dir, "master_process.log")
        # We use 'a' (append) so we don't accidentally wipe it, 
        # though 'w' is fine for a fresh run.
        m_log = open(master_log_path, 'w', buffering=1)
        sys.stdout = m_log
        sys.stderr = m_log
        
        print(f"Master process started on Rank {rank}")
        print(f"Directory: {log_dir}")
        print("-" * 40)
        # loading the dataframe
        data_frame = pd.read_csv(TARGET_PATH, low_memory=False)
        # sampling the stars for injection recovery
        if injrec:
            data_frame=data_frame.sample(n=injrec).reset_index(drop=True)
            data_frame["ir_pkl_path"] = [f"{DATA_DIR}/{row.TICID}/ir_{row.sectors}_120.pkl" for _, row in data_frame.iterrows()]
            data_frame[['TICID','sectors', 'ir_pkl_path']].to_csv(f"{log_dir}/ir_star_list.csv", index=False)
        # chunking the data to be given to each worker.
        # --- CRITICAL FIX HERE ---
        # Convert the DataFrame into a native list of pure Python dictionaries.
        # This completely unhooks the data from the problematic NumPy/Pandas C-API layouts.
        list_of_records = data_frame.to_dict(orient='records')
        
        # Split the native Python list into chunks across ranks
        chunks = np.array_split(list_of_records, size)
        
        # Convert NumPy array fragments back to regular python lists of dicts
        chunks = [list(chunk) for chunk in chunks]
        print("Chunks split completed successfully via native Python objects.")
    else:
        chunks = None

    # sending the data to each worker core
    my_chunk = comm.scatter(chunks, root=0)

    terminal_stdout = sys.stdout

    for star_row in my_chunk:
        TIC = star_row['TICID']
        sector = star_row['sectors']
        cadence = 120
        period = star_row['sector_periods']
        distance = star_row['distance']
        if os.environ.get("USER") == "krohan":
            PATH_COLUMN = "path_to_lc"
        else:
            PATH_COLUMN = "path_to_local_lc"
        try:
            lc_path = star_row[PATH_COLUMN]
        except:
            lc_path=None
        lc_path=None # download lightcurve in all the cases

        log_file_path = os.path.join(log_dir, f"TIC{TIC}_S{sector:03d}_C{cadence}.log")

        with open(log_file_path, 'w') as f:
            sys.stdout = f
            sys.stderr = f

            start_time=time.time()
            try:
                mem_before = get_memory_usage()
                if injrec:
                    tess_pipeline_mpi(tic=TIC, data_dir=DATA_DIR, lc_dir=lc_path,redo=rerun, injrec=1, cadence=cadence, sector=sector, period=period, distance=distance, calc_energy=True)
                    stop_time=time.time()
                else:
                    tess_pipeline_mpi(tic=TIC, data_dir=DATA_DIR, lc_dir=lc_path, redo=rerun, injrec=0, cadence=cadence, sector=sector, period=period, distance=distance, calc_energy=True)
                    stop_time=time.time()
                mem_after = get_memory_usage()
            except Exception as e:
                # Any crash in your function will be caught here and logged
                mem_after = get_memory_usage()
                print(f"CRITICAL ERROR in pipeline: {e}")
                traceback.print_exc()
                stop_time=time.time()
            print(f"Runtime::{stop_time-start_time:.2f} s")
            print(f"Memory usage::{mem_after-mem_before:.2f} MB")

            # RESET: Point output back to the appropriate 'Home'
            if rank == 0:
                if m_log and not m_log.closed:
                    sys.stdout = m_log
                    sys.stderr = m_log
                else:
                    sys.stdout = terminal_stdout
                    sys.stderr = terminal_stdout
            else:
                sys.stdout = terminal_stdout
                sys.stderr = terminal_stdout

    # --- 6. FINAL SYNCHRONIZATION ---
    comm.Barrier()
        
    if rank == 0:
        print("-" * 50)
        print(f"Master: All {size} ranks have completed their tasks.")
        print(f"Master: Run finished at {time.ctime()}")
        
        # Return to terminal for the final goodbye
        sys.stdout = terminal_stdout
        sys.stderr = terminal_stdout

        # Final cleanup of Master Log
        m_log.close()
        
        print(f"\n>>> RUN COMPLETE. Results saved in: {log_dir}")

if __name__ == "__main__":
    main()