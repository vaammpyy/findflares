import multiprocessing as mp
from datetime import datetime
from contextlib import redirect_stdout
from time import time
import sys
from astropy import units as u

from findflares.lc_class import *
from findflares.imports import *

def tess_pipeline(tic, data_dir, redo=True, injrec=0, input_cadence=0, input_sector=0, calc_energy=True):
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
    calc_energy : bool, optional
        If true then calculated energy of the flares, by default true.
    
    Output
    -------
    Generates plots and stores pickles.
    """
    print(f"RE-RUN::{redo}")
    print(f"Inj-Rec::{injrec}")
    print("STATUS::STARTED")
    print("###################")
    print(f"META::TIC={tic}", flush=True)
    cad_list=[20, 120]
    print(f"Searching for observations with CADENCE: {cad_list}")
    error=None
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
                    print(f"META::SECTOR={sector}\nMETA::CADENCE={cad}", flush=True)
                    print("Already exists.")
                else:
                    print(f"META::SECTOR={sector}\nMETA::CADENCE={cad}", flush=True)
                    # pipeline(TIC, sector, cad)
                    print("****************")
                    lc=TESSLC(tic, data_dir+"/"+str(TIC))
                    # lc.download_lc(sector=sector, cadence=cad, segment=True, clean=True)
                    try:
                        lc.download_lc(sector=sector, cadence=cad, clean=True)
                    except HTTPError:
                        error="HTTPError"
                        print("Download failed. Downloading next sector.")
                        continue
                    except ConnectionError:
                        error="ConnectionError"
                        print("ConnectionError, failed to connect to the server.")
                        continue
                    lc.detrend()
                    lc.findflares()
                    lc.flare_energy(calc_energy=calc_energy)
                    if injrec:
                        print("Injection recovery test started.")
                        irec=InjRec(lc)
                        for k in range(injrec):
                            irec.run_injection_recovery(run=k+1, plot=False)
                        print("Injection recovery test completed.")
                        plot_ir_results(irec, mode='rec_frac', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_sa_ampl', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_sa_fwhm', save_fig=True)
                        if irec.star.dist is not None:
                            plot_ir_results(irec, mode='erg_comp', save_fig=True)
                            plot_ir_results(irec, mode='fp', save_fig=True)
                            plot_ir_results(irec, mode='rec_frac_erg', save_fig=True)
                        else:
                            print("PIPELINE::COMMENT::Distance not found skipping energy plots.")
                        plot_ir_results(irec, mode='rec_frac_spot_amplitude', save_fig=True)
                        plot_ir_results(irec, mode='inj_spot_amplitude', save_fig=True)
                        plot_ir_results(irec, mode='inj_spot_amplitude', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_spot_amplitude_binned', save_fig=True)
                        # irec.pickleObj()
                    else:
                        lc.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="flare_overlay", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="flare_model_overlay", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="model_overlay", save_fig=True)
                        # lc.pickleObj()
        else:
            print("No data found!")
    except HTTPError:
        error="HTTPError"
        print("HTTPError, failed to fetch data.")
    except ConnectionError:
        error="ConnectionError"
        print("ConnectionError, failed to connect to the server.")
    finally:
        if injrec:
            irec.pickleObj()
        else:
            lc.pickleObj()
        if sys.exc_info()[0] is None and error is None:
            print("****************")
            print("STATUS::SUCCESS!", flush=True)
        else:
            print("****************")
            print(f"STATUS::{sys.exc_info()[0]} {error}::FAILURE :(", flush=True)
    
def tess_pipeline_mpi(tic, data_dir, redo=True, injrec=0, cadence=0, sector=0, period=None, distance=None, calc_energy=True):
    """
    Runs pipeline for TESS Lightcurves, using mpi4py interface.

    Runs pipeline for TESS Lightcurves, and stores output in data_dir. Uses mpi4py interface.

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
    cadence : int, optional
        Cadence of the observation for analysis, by default value is 0
        pipeline runs over both 120s and 20s data.
    sector : int, optional
        Sector of the observation for analysis, by default value is 0
        pipeline runs over all avialable sectors.
    period : float, optional
        Rotation period of the star if known.
    distance : float, optional 
        Distance to the star if known, in parsecs. If not known then pipeline will attempt to fetch the distance.
    calc_energy : bool, optional
        If true then calculated energy of the flares, by default true.
    
    Output
    -------
    Generates plots and stores pickles.
    """
    print(f"RE-RUN::{redo}")
    print(f"Inj-Rec::{injrec}")
    print("STATUS::STARTED")
    print("###################")
    print(f"META::TIC={tic}", flush=True)
    cad_list=[20, 120]
    print(f"Searching for observations with CADENCE: {cad_list}")
    error=None
    already_exist=False
    try:
        if cadence and sector:
            search_result = [(tic, sector, cadence)]
        else:
            search_result = search_lightcurve(tic, cadence=cad_list, ret_list=True)
        if search_result:
            for result in search_result:
                TIC, sector, cad = result
                # checking if the pipeline has already run or not
                if os.path.isfile(f"{data_dir}/{TIC}/{sector}_{cad}.pkl") and redo==False:
                    print(f"META::SECTOR={sector}\nMETA::CADENCE={cad}", flush=True)
                    print("Already exists.")
                    already_exist=True
                else:
                    print(f"META::SECTOR={sector}\nMETA::CADENCE={cad}", flush=True)
                    # pipeline(TIC, sector, cad)
                    print("****************")
                    lc=TESSLC(tic, data_dir+"/"+str(TIC))
                    # lc.download_lc(sector=sector, cadence=cad, segment=True, clean=True)
                    try:
                        lc.download_lc(sector=sector, cadence=cad, clean=True)
                    except HTTPError:
                        error="HTTPError"
                        print("Download failed. Downloading next sector.")
                        continue
                    except ConnectionError:
                        error="ConnectionError"
                        print("ConnectionError, failed to connect to the server.")
                        continue
                    lc.star.dist=distance*u.pc
                    if period:
                        lc.detrend(period=period)
                    else:
                        lc.detrend()
                    lc.findflares()
                    lc.flare_energy(calc_energy=calc_energy)
                    if injrec:
                        print("Injection recovery test started.")
                        irec=InjRec(lc)
                        for k in range(injrec):
                            irec.run_injection_recovery(run=k+1, plot=False)
                        print("Injection recovery test completed.")
                        plot_ir_results(irec, mode='rec_frac', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_sa_ampl', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_sa_fwhm', save_fig=True)
                        if irec.star.dist is not None:
                            plot_ir_results(irec, mode='erg_comp', save_fig=True)
                            plot_ir_results(irec, mode='fp', save_fig=True)
                            plot_ir_results(irec, mode='rec_frac_erg', save_fig=True)
                        else:
                            print("PIPELINE::COMMENT::Distance not found skipping energy plots.")
                        plot_ir_results(irec, mode='rec_frac_spot_amplitude', save_fig=True)
                        plot_ir_results(irec, mode='inj_spot_amplitude', save_fig=True)
                        plot_ir_results(irec, mode='inj_spot_amplitude', save_fig=True)
                        plot_ir_results(irec, mode='rec_frac_spot_amplitude_binned', save_fig=True)
                        # irec.pickleObj()
                    else:
                        lc.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="flare_overlay", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="flare_model_overlay", show_flares=True, show_transits=True, save_fig=True)
                        lc.plot(mode="model_overlay", save_fig=True)
                        # lc.pickleObj()
        else:
            print("No data found!")
    except HTTPError:
        error="HTTPError"
        print("HTTPError, failed to fetch data.")
    except ConnectionError:
        error="ConnectionError"
        print("ConnectionError, failed to connect to the server.")
    finally:
        if injrec and already_exist == False:
            irec.pickleObj()
        elif already_exist == False:
            lc.pickleObj()
        if sys.exc_info()[0] is None and error is None:
            print("****************")
            print("STATUS::SUCCESS!", flush=True)
        else:
            print("****************")
            print(f"STATUS::{sys.exc_info()[0]} {error}::FAILURE :(", flush=True)
    
def spawn_pipeline_process(candidate, data_dir, redo, injrec, output_dir, process_index):
    """
    Spawns a pipeline process.

    Spawns a pipeline process for multicore analysis, for single core process spawns process
    sequentially.

    Parameters
    ----------
    candidate : tuple (int, int, int)
        Tuple with (TIC, sector, cadence).
    data_dir : str
        Directory to store the data.
    redo : bool
        If True re-runs the pipeline for existing data.
    injrec : int
        If '1' then performs injection recovery tests.
    output_dir : str
        Directory to store the output from the runs.
    process_index : int
        Index of the process initiated to avoid overwriting of output files
    """
    process_id=mp.current_process().pid
    ID, sector, cad=candidate

    output_file = os.path.join(output_dir, f"{process_id}_{process_index}.out")

    track_job_file=os.path.join(data_dir+f"/{ID}","job.out")

    try:
        os.mkdir(f"{data_dir}/{ID}")
    except OSError:
        pass


    with open(track_job_file, 'a') as f:
        with redirect_stdout(f):
            print(f"{output_dir}/{process_id}_{process_index}.out")

    with open(output_file, 'w') as f:
        with redirect_stdout(f):
            print(f"PID:{process_id}_{process_index}\nCANDIDATE:{candidate}\nREDO:{redo}\nINJREC:{injrec}\n")
            t_start=time()
            tess_pipeline(ID, data_dir, redo, injrec, cad, sector)
            t_stop=time()
            print(f"TIME:{t_stop-t_start:.2f}")

def run_pipeline_py(candidate_file, telescope, data_dir, output_dir, CPU_CORES=1, redo=True, injrec=0):
    """
    Runs the pipeline

    General module to run a pipeline for a specific target list on specified number of cores.

    Parameters
    ----------
    candidate_file : str
        Path to the file having list of all candidates
    telescope : str
        Telescope name for pipeline, kepler reduction will be added
    data_dir : str
        Path to the directory for storing data.
    output_dir : str
        Path to the directory for storing outputs.
    CPU_CORES : int, optional
        Number of CPU cores to run the process, by default 1.
    redo : bool, optional
        If True re-runs the pipeline for existing data, by default True.
    injrec : bool, optional
        If True runs injection recovery tests.
    """
    candidate_arr=[]
    with open(candidate_file, 'r') as file:
        for line in file:
            try:
                ID, sector, cad=line.split(',')
                candidate_arr.append((int(ID), int(sector), int(cad)))
            except ValueError:
                ID=line
                sector=0
                cad=0
                candidate_arr.append((int(ID), int(sector), int(cad)))
    
    time = datetime.now().strftime("%H%M_%d%m%Y")
    job_id=f"job_{time}"
    output_dir = f"{output_dir}/{job_id}"

    try:
        os.mkdir(output_dir)
    except OSError:
        pass

    with mp.Pool(processes=CPU_CORES) as pool:
        pool.starmap(spawn_pipeline_process, [(candidate, data_dir, redo, injrec, output_dir, index) for index, candidate in enumerate(candidate_arr)])

# here I'll add a separate module for running the pipeline for slurm process
# This new process will take in a TIC derived from a TIC list and then 
# start the TESS_pipeline module for that TIC.