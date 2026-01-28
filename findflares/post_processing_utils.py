import os
from glob import glob
import numpy as np

from .lc_class import *

def _return_run_summary(file_path):
    """
    This module reads the output file and returns the run status.

    This module reads the output file and returns the run status along
    with the relevant meta data of the star such as TIC, sector and cadence.

    Parameters
    ----------
    file_path : str
        Path to the output file.
    """
    meta = {}
    with open(file_path) as f:
        for line in f:
            if line.startswith("META::"):
                key, val = line.strip().split("::")[1].split("=")
                meta[key] = val
            if line.startswith("STATUS::"):
                key, val = line.strip().split("::")
                meta[key] = val
        status=-1
        if meta["STATUS"] == "COMPLETED":
            status=0
    return  meta["TIC"], meta["SECTOR"], meta["CADENCE"], status

def _log_failed_runs(run_summary_dict, file_path):
    """
    This module logs the failed runs into a file.

    This module logs the stars with TIC, sector and cadence into a file. The
    generated file can then be directly used for the next run.

    Parameters
    ----------
    run_summary_dict : dict
        Dictionary which stores the run status and meta data for all the stars.
    file_path : str
        Path to save the failed stars file.
    """
    failed_stars_dict = {"TIC":[], "sector":[], "cadence":[]}
    for i,status in enumerate(run_summary_dict["status"]):
        if status == -1:
            failed_stars_dict["TIC"].append(run_summary_dict["TIC"][i])
            failed_stars_dict["sector"].append(run_summary_dict["sector"][i])
            failed_stars_dict["cadence"].append(run_summary_dict["cadence"][i])
    df=pd.DataFrame(failed_stars_dict)
    df.to_csv(file_path, index=False, header=False)

def log_results(output_dir):
    """
    This module creates a compiled result table for the further statistical analysis.

    This module creates a compiled run status table along with the meta data for further statistical analysis.

    Parameters
    ----------
    output_dir : str
        Path to the output directory storing the run files.
    """
    output_file_list = glob(f"{output_dir}/TIC*")

    run_summary_dict={"TIC":[],
                    "sector":[],
                    "cadence":[],
                    "status":[]}

    for output_file in output_file_list:
        tic, sector, cadence, status = _return_run_summary(output_file)
        run_summary_dict["TIC"].append(tic)
        run_summary_dict["sector"].append(sector)
        run_summary_dict["cadence"].append(cadence)
        run_summary_dict["status"].append(status)

    _log_failed_runs(run_summary_dict, f"{output_dir}/failed_stars.csv")

def _extract_parameters(TESSLC_path):
    """
    This module extracts parameters.

    This module reads the pickle file and extracts both the stellar and flare
    parameters.

    Parameters
    ----------
    TESSLC_path : str
        Path to the tess lightcurve pickle path.
    
    Returns
    -------
    tic : int
        TIC ID.
    sector : int
        Observation sector.
    cadence : float
        Observation cadence.
    tess_mag : float
        TESS magnitude of the star.
    n_flare : int
        Total number of flares detected.
    Teff : float
        Effective temperature of the star.
    radius : float
        Radius of the star in R_sun.
    period : float
        Rotation period (GLS) of the star if star is found to be rotating, else np.nan.
    period_GP : float
        Rotation period (GP) of the star if star is found to be rotating, else np.nan.
        Period here is defined as the average of the period over all segments.
    range_period_gp : float
        Difference between the maximum period and minimum period of all the segments.
    [--]*n_flares : list
        List of stellar parameters for the flare table.
    t_peak : list
        Flare peak time.
    amplitude : list
        Amplitude of the flare.
    energy : list
        Energy of the flare.
    duration : list
        Duration of the flare.
    equivalent_duration : list
        Equivalent duration of the flare.
    spot_amplitude : list
        Spot amplitude of the flare.
    flare_number : list
        Time ordered numbering of the flares.
    """
    lc=ff.loadpickle_path(TESSLC_path)

    # stellar parameters
    tic = lc.TIC
    sector = lc.inst.sector
    cadence = lc.inst.cadence * 24*3600
    tess_mag = lc.star.tess_mag
    n_flares = len(lc.flares["t_start"])
    Teff = lc.star.teff
    radius = lc.star.rad

    if lc.star.prot:
        period = lc.star.prot 
        period_GP = np.average(lc.star.prot_GP)
        range_period_gp = max(lc.star.prot_GP) - min(lc.star.prot_GP)
    else:
        period = np.nan
        period_GP = np.nan
        range_period_gp = np.nan

    # flare parameters
    t_peak = lc.flares["t_peak"]
    amplitude = lc.flares["amplitude"]
    energy = lc.flares["energy"]
    duration = lc.flares["duration"]
    equivalent_duration = lc.flares["equi_duration"]
    spot_amplitude = lc.flare["spot_amplitude"]
    flare_number = [i+1 for i in range(n_flares)]
    

    return tic, sector, cadence, tess_mag, n_flares, Teff, radius, period, period_GP, range_period_gp, [tic]*n_flares, [sector]*n_flares, [cadence]*n_flares, [tess_mag]*n_flares, [Teff]*n_flares, [radius]*n_flares, [period]*n_flares, [period_GP]*n_flares, [range_period_gp]*n_flares, t_peak, amplitude, energy, duration, equivalent_duration, spot_amplitude, flare_number
        
def get_compiled_catalog(flare_dir, flare_catalog_file_path, star_catalog_file_path):
    """
    This module creates the flare and stellar properties of the successful run.
    """
    pickle_path_list = glob(f"{flare_dir}/*/*.pkl")

    flare_table = Table()
    stellar_table = Table()

    
    # list of stellar properties
    tic_list = []
    sector_list = []
    cadence_list = []
    tess_mag_list = []
    n_flares_list = []
    Teff_list = []
    radius_list = []
    period_list = []
    period_GP_list = []
    range_GP_period_list = []

    # list of flare properties
    tic_list_flare = []
    sector_list_flare = []
    cadence_list_flare = []
    tess_mag_list_flare = []
    Teff_list_flare = []
    radius_list_flare = []
    period_list_flare = []
    period_GP_list_flare = []
    range_GP_period_list_flare = []
    t_peak_list_flare = []
    amplitude_list_flare = []
    energy_list_flare = []
    duration_list_flare = []
    equivalent_duration_list_flare = []
    spot_amplitude_list_flare = []
    flare_number_list_flare = []

    for pickle_path in pickle_path_list:
        tic, sector, cadence, tess_mag, n_flares, Teff, radius, period, period_GP, range_period_gp, tic_flare, sector_flare, cadence_flare, tess_mag_flare, Teff_flare, radius_flare, period_flare, period_GP_flare, range_period_gp_flare, t_peak_flare, amplitude_flare, energy_flare, duration_flare, equivalent_duration_flare, spot_amplitude_flare, flare_number_flare =  _extract_parameters(pickle_path)

        # loggin stellar parameters.
        tic_list.append(tic)
        sector_list.append(sector)
        cadence_list.append(cadence)
        tess_mag_list.append(tess_mag)
        n_flares_list.append(n_flares)
        Teff_list.append(Teff)
        radius_list.append(radius)
        period_list.append(period)
        period_GP_list.append(period_GP)
        range_GP_period_list.append(range_period_gp)

        # logging flare parameters.
        tic_list_flare.extend(tic_flare)
        sector_list_flare.extend(sector_flare)
        cadence_list_flare.extend(cadence_flare)
        tess_mag_list_flare.extend(tess_mag_flare)
        Teff_list_flare.extend(Teff_flare)
        radius_list_flare.extend(radius_flare)
        period_list_flare.extend(period_flare)
        period_GP_list_flare.extend(period_GP_flare)
        range_GP_period_list_flare.extend(range_period_gp_flare)
        t_peak_list_flare.extend(t_peak_flare)
        amplitude_list_flare.extend(amplitude_flare)
        energy_list_flare.extend(energy_flare)
        duration_list_flare.extend(duration_flare)
        equivalent_duration_list_flare.extend(equivalent_duration_flare)
        spot_amplitude_list_flare.extend(spot_amplitude_flare)
        flare_number_list_flare.extend(flare_number_flare)
    
    # logging stellar properties
    stellar_table["TIC"] = tic_list
    stellar_table["sector"] = sector_list
    stellar_table["cadence"] = cadence_list
    stellar_table["tess_mag"] = tess_mag_list
    stellar_table["n_flares"] = n_flares_list
    stellar_table["Teff"] = Teff_list
    stellar_table["radius"] = radius_list
    stellar_table["period"] = period_list
    stellar_table["period_GP"] = period_GP_list
    stellar_table["range_GP_period"] = range_GP_period_list

    # logging flare parameters
    flare_table["TIC"] = tic_list_flare
    flare_table["sector"] = sector_list_flare
    flare_table["cadence"] = cadence_list_flare
    flare_table["tess_mag"] = tess_mag_list_flare
    flare_table["Teff"] = Teff_list_flare
    flare_table["radius"] = radius_list_flare
    flare_table["period"] = period_list_flare
    flare_table["period_GP"] = period_GP_list_flare
    flare_table["range_GP_period"] = range_GP_period_list_flare
    flare_table["t_peak"] = t_peak_list_flare
    flare_table["amplitude"] = amplitude_list_flare
    # flare_table["energy"] = energy_list_flare
    flare_table["duration"] = duration_list_flare
    flare_table["equivalent_duration"] = equivalent_duration_list_flare
    flare_table["spot_amplitude"] = spot_amplitude_list_flare
    flare_table["flare_number"] = flare_number_list_flare

    stellar_table.write(f"{star_catalog_file_path}/stellar_catalog.txt",
                        format='ascii.mrt')
    flare_table.write(f"{flare_catalog_file_path}/flare_catalog.txt",
                        format='ascii.mrt')