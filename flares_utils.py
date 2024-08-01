import numpy as np
from FINDflare_dport import FINDflare
from aflare import aflare, aflare1
import pdb
from scipy.integrate import simpson
import astropy.units as u
import astropy.constants as const
from misc import get_dist_gaia, get_dist_tess
from random import sample, uniform

def _merge_flares(start_indices, stop_indices, close_th):
    """
    Merges the nearby flares.

    Parameters
    ----------
    start_indices : ndarray
        Start indices of the flares.
    stop_indices : ndarray
        Stop indices of the flares.
    close_th : int
        Threshold to merge the flares, typically taken to be 9 data points as it corresponds to 180s mentioned in section 3.2 Davenport 2016.
    
    Returns
    -------
    new_start_indices : ndarray
        Start indices of the flares after merging.
    new_stop_indices : ndarray
        Stop indices of the flares after merging.
    """
    # Check if the input arrays are valid
    if len(start_indices) != len(stop_indices):
        raise ValueError("Start and stop indices arrays must have the same length")
    
    # Initialize the lists for new start and stop indices
    new_start_indices = []
    new_stop_indices = []
    
    # Initialize the first event
    current_start = start_indices[0]
    current_stop = stop_indices[0]
    
    for i in range(1, len(start_indices)):
        next_start = start_indices[i]
        next_stop = stop_indices[i]
        
        # Check if the difference between the current stop and next start is less than or equal to close_threshold 
        if next_start - current_stop <= close_th:
            # Merge events by updating the current stop to the next stop
            current_stop = next_stop
        else:
            # Append the current event to the new lists
            new_start_indices.append(current_start)
            new_stop_indices.append(current_stop)
            # Update the current event to the next event
            current_start = next_start
            current_stop = next_stop
    
    # Append the last event to the new lists
    new_start_indices.append(current_start)
    new_stop_indices.append(current_stop)
    
    return new_start_indices, new_stop_indices

def _include_tail(data, start_indices, stop_indices, sig_lvl, transit=False):
    """
    Includes the leading and lagging tail around a flare,  to avoid loosing the detectable part of a flare.

    To avoid loosing the detectable part of flare (tails), we look for points near the flare which are above the
    threshold deviation above the mean until one of them drops below it and extend the flare start and stop times
    upto that point.

    Parameter
    ---------
    data : ndarray
        Flux array.
    start_indices : ndarray
        Start indices of the flare.
    stop_indices : ndarray
        Stop indices of the flare.
    sig_lvl : float
        Threshold deviation for the tail.
    transit : bool, optional
        If True includes tail for a transit signal, by default False.
    
    Returns
    -------
    new_start_indices : ndarray
        Start indices of the flares after including the tail.
    new_stop_indices : ndarray
        Stop indices of the flares after including the tail.
    """
    # Calculate mean and standard deviation of the data
    mean = np.mean(data)
    std_dev = np.std(data)

    new_start_indices = []
    new_stop_indices = []
    
    # Determine the threshold
    if transit:
        threshold = mean - sig_lvl * std_dev
        
        for start, stop in zip(start_indices, stop_indices):
            new_start = start
            new_stop = stop
            
            # Adjust start index backwards
            for i in range(start - 1, -1, -1):
                if data[i] < threshold:
                    # new_start = i-100
                    new_start = int(i*0.995)
                else:
                    break
            
            # Adjust stop index forwards
            for i in range(stop + 1, len(data)):
                if data[i] < threshold:
                    # new_stop = i+100
                    new_stop = int(i*1.005)
                else:
                    break
            
            new_start_indices.append(new_start)
            new_stop_indices.append(new_stop)
        return new_start_indices, new_stop_indices
    else:
        threshold = mean + sig_lvl * std_dev
        
        for start, stop in zip(start_indices, stop_indices):
            new_start = start
            new_stop = stop
            
            # Adjust start index backwards
            for i in range(start - 1, -1, -1):
                if data[i] > threshold:
                    new_start = i
                else:
                    break
            
            # Adjust stop index forwards
            for i in range(stop + 1, len(data)):
                if data[i] > threshold:
                    new_stop = i
                else:
                    break
            
            new_start_indices.append(new_start)
            new_stop_indices.append(new_stop)
    
        return new_start_indices, new_stop_indices

def find_flare(obj, find_transit=False):
    """
    Finds flares in the lightcurve.

    Finds flares in the lightcurve object using the modified critrion described in Chang et al. 2015 and Davenport 2016.
    Start and stop times the flares are also extended by merging the close flares and tracking the tail as described in
    Davenport 2016 and Ilin and Poppenhaeger 2022.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    find_transit : bool, optional
        True for finding transit.

    Attributes
    ----------
    obj.lc.flare : dict
        Flare dictionary, {'start', 'stop', 'mask'}, note that the flare mask is negative i.e 0 means flare point.
    obj.lc.flare_run : bool
        Flare algorithm run check.
    obj.lc.transit : dict, optional (defined when find_transit=True)
        Transit dictionary, {'start', 'stop', 'mask'}, note that the transit mask is negative i.e 0 means transit point.
    obj.lc.transit_run : bool
        Transit algorithm run check.
    """
    obj.lc.flare_run=True
    if find_transit:
        obj.lc.transit_run=True
    data=obj.lc.detrended
    flux=data['flux']
    flux_err=data['flux_err']
    start, stop=FINDflare(flux, flux_err, N1=3, N2=1, N3=3, avg_std=True, std_window=5)
    # checking if flares are found
    if len(start)>0: 
        if int(obj.inst.cadence*24*3600) == 20:
            close_th=18
        if int(obj.inst.cadence*24*3600) == 120:
            close_th=3
        start, stop= _include_tail(flux, start, stop, sig_lvl=1)
        start, stop=_merge_flares(start, stop, close_th=close_th)
        #making a negative flare mask
        # flare_mask=np.ones(len(obj.lc.segment), dtype=bool)
        flare_mask=np.ones(len(obj.lc.full['time']), dtype=bool)

    for i in range(len(start)):
        flare_mask[start[i]:stop[i]+1]=0

    if len(start)>0:
        obj.lc.flare={'start': start,
                    'stop': stop,
                    'mask': flare_mask}
    else:
        obj.lc.flare={'start': start,
                    'stop': stop,
                    'mask': np.array([])}

    t_start, t_stop= [], []
    if find_transit:
        t_start, t_stop=FINDflare(flux, flux_err, N1=3, N2=1, N3=50, avg_std=False, std_window=360, find_transit=True)
        t_start, t_stop= _include_tail(flux, t_start, t_stop, sig_lvl=0.5, transit=True)
        transit_mask=np.ones(len(obj.lc.segment), dtype=bool)
        for i in range(len(t_start)):
            transit_mask[t_start[i]:t_stop[i]+1]=0
        obj.lc.transit={'start': t_start,
                    'stop': t_stop,
                    'mask': transit_mask}
    elif obj.lc.transit_run:
        pass
    else:
        obj.lc.transit={'start':t_start,
                    'stop': t_stop,
                    'mask': np.array([])}

def get_flare_param(obj):
    """
    Calculates important flare parameters.

    Calculates the following flare parameters and appends it to self.flares dictionary.

    start time of the flare [mjd] : t_start
    stop time of the flare [mjd] : t_stop
    start index of the flare [array index] : i_start
    stop index of the flare [array index] : i_stop
    amplitude of the flare [e/s] : amplitude, calculation is based on eq 3, Hawley et al. 2014.
    duration of the flare [s] : duration

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.

    Attributes
    ----------
    obj.flares : dict
        {"t_start":, [mjd]
        "t_peak":, [mjd]
        "t_stop":, [mjd]
        "i_start":,
        "i_stop":,
        "amplitude":, [e/s]
        "duration":, [s]}
    """
    print("Estimating flare parameters.")
    time=obj.lc.detrended['time']
    flux_detrended=obj.lc.detrended['flux']
    flux_model=obj.lc.model['flux']
    flux=obj.lc.full['flux']
    t_start=obj.lc.flare['start']
    t_stop=obj.lc.flare['stop']

    for i in range(len(t_start)):
        start, stop=_get_flare_tstart_tstop(time, flux_detrended, t_start[i], t_stop[i])
        obj.flares['t_start'].append(time[start])
        obj.flares['t_stop'].append(time[stop])
        obj.flares["i_start"].append(start)
        obj.flares["i_stop"].append(stop)
        flux_peak_index=np.argmax(flux_detrended[start:stop+1])
        amplitude=flux_detrended[start:stop+1][flux_peak_index]
        obj.flares['amplitude'].append(amplitude)
        dur=(time[stop]-time[start])*24*3600
        obj.flares['duration'].append(dur)
        obj.flares['t_peak'].append(time[start:stop+1][flux_peak_index])

def get_ED(obj):
    """
    Calculates Equivalent duration of the flare.

    Calculates Equivalent duration of the flare as stated in eq 2 section 3, Gershberg - 1972 and
    eq 3 in section 3 of Hunt-Walker et al. 2012.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    
    Attributes
    ----------
    obj.flares['equi_duration'] : list [seconds]
        Equivalent duration of all the flares.
    
    Notes
    -----
    This function assumes that get_flare_param has already been run before.
    """
    print("Estimating flare ED.")
    time=obj.lc.detrended['time']
    flux_detrended=obj.lc.detrended['flux']
    flux_model=obj.lc.model['flux']

    y=flux_detrended/flux_model

    flares_dict=obj.flares

    n_flares=len(flares_dict['t_start'])

    for i in range(n_flares):
        in_flare_time=time[flares_dict['i_start'][i]: flares_dict["i_stop"][i]+1]
        in_flare_flux_norm=y[flares_dict['i_start'][i]: flares_dict["i_stop"][i]+1]
        ed=simpson(in_flare_flux_norm, x=in_flare_time)*24*3600
        obj.flares['equi_duration'].append(ed)

def _get_flare_tstart_tstop(time, flux, t_start, t_stop, rel_diff_th=0.01):
    """
    Changes the flare start and stop until addition of new points
    does not change the integral of luminosity under the flare (relative difference<0.01).
    Adapted from section 3.2, Diamond-Lowe et al. 2021.

    Parameters
    ----------
    time : np.array
        Time array of the entire lightcurve.
    flux : np.array
        Flux array of the entire lightcurve.
    t_start : int
        Start index of flare as found by modified FindFlare.
    t_stop : int
        Stop index of flare as found by modified FindFlare.
    rel_diff_th : float, optional
        Relative difference threshold for the next interation, by default 0.01.
    
    Returns
    -------
    t_start : int
        Modified start index of the flare.
    t_stop : int
        Modified stop index of the flare.
    """
    # evaluating the stop time of the flare.
    change_stop=True
    while change_stop:
        in_flare_time_i=time[t_start:t_stop+1]
        in_flare_flux_i=flux[t_start:t_stop+1]
        integral_i=simpson(in_flare_flux_i, x=in_flare_time_i)
        t_stop_i_1=t_stop+1
        in_flare_time_i_1=time[t_start:t_stop_i_1+1]
        in_flare_flux_i_1=flux[t_start:t_stop_i_1+1]
        integral_i_1=simpson(in_flare_flux_i_1, x=in_flare_time_i_1)
        rel_diff=(integral_i_1-integral_i)/integral_i
        if rel_diff>rel_diff_th:
            change_stop=True
            t_stop=t_stop_i_1
        else:
            change_stop=False

    change_start=True
    while change_start:
        in_flare_time_i=time[t_start:t_stop+1]
        in_flare_flux_i=flux[t_start:t_stop+1]
        integral_i=simpson(in_flare_flux_i, x=in_flare_time_i)
        t_start_i_1=t_start-1
        in_flare_time_i_1=time[t_start_i_1:t_stop+1]
        in_flare_flux_i_1=flux[t_start_i_1:t_stop+1]
        if t_start_i_1<0:
            t_start=0
            break
        integral_i_1=simpson(in_flare_flux_i_1, x=in_flare_time_i_1)
        rel_diff=(integral_i_1-integral_i)/integral_i
        if rel_diff>rel_diff_th:
            change_start=True
            t_start=t_start_i_1
        else:
            change_start=False

    return t_start, t_stop

def get_flare_energies(obj):
    """
    Calculates flare energies

    Calculates flare energies using the eq 1 section 3.2 in Diamond-lowe et al. 2021

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    
    Attributes
    ----------
    obj.flares['energy'] : Energy of the flares
    """
    TIC=obj.TIC
    ra=obj.star.ra
    dec=obj.star.dec
    dist_pc=get_dist_gaia(ra, dec)
    if dist_pc is None:
        dist_pc=get_dist_tess(TIC)
    else:
        print("Distance not found.")
        print("Flare energy calculation skipped.")
        return
    dist_cm=dist_pc.to(u.cm)
    obj.star.dist=dist_pc
    flux_detrended=obj.lc.detrended['flux']
    time=obj.lc.detrended['time']

    flares_dict=obj.flares

    n_flares=len(flares_dict['t_start'])

    for i in range(n_flares):
        in_flare_time=time[flares_dict['i_start'][i]: flares_dict["i_stop"][i]+1]
        in_flare_flux=flux_detrended[flares_dict['i_start'][i]: flares_dict["i_stop"][i]+1]
        e_count=simpson(in_flare_flux, x=in_flare_time)*24*3600
        lambda_mean=7452.64*u.angstrom

        h=const.h
        c=const.c

        h_cgs = h.to(u.erg * u.s)
        c_cgs = c.to(u.cm / u.s)
        energy_per_electron = h_cgs * c_cgs / lambda_mean.to(u.cm)

        # Calculate total electron energy and energy
        tot_electron_energy = energy_per_electron * e_count
        # calculates the energy, 86.6 cm2 is aperture area.
        energy = 4 * np.pi * dist_cm**2 * tot_electron_energy/86.6

        obj.flares['energy'].append(energy.value)

def replace_flares_w_gaussian_noise_and_clean_attr(obj):
    """
    Replaces flares with gaussian noise and cleans attributes.

    Replaces flares with gaussian noise derived from the data, and removes all model information.

    Parameters
    ----------
    obj : InjRec
        Injection Recovery object
    
    Attributes
    ----------
    obj.lc.full : dict
        Lightcurve dictionary storing lightcurve information.
    obj.lc.detrended : dict
        Detrended lightcurve model, assigned None
    obj.lc.detrend_scheme : str
        Detrending scheme used, assigned None
    obj.lc.flare : dict
        Flare information dict, assigned None
    obj.lc.flare_run : bool
        Find flare run check, assigned None
    obj.lc.transit : dict
        Transit information dict, assigned None
    obj.lc.transit_run : bool
        Find transit run check, assigned None
    """
    lc_dict=obj.lc.detrended

    flux_mean=np.mean(lc_dict['flux'])
    flux_std=np.std(lc_dict['flux'])

    flares_dict=obj.flares

    n_flares=len(flares_dict['i_start'])
    for i in range(n_flares):
        start=flares_dict['i_start'][i]
        stop=flares_dict['i_stop'][i]

        n_samples=stop-start+1

        obj.lc.detrended['flux'][start:stop+1]=np.random.normal(flux_mean, flux_std, n_samples)

    obj.lc.full['flux']=obj.orig_lc.model['flux']+obj.lc.detrended['flux']
    obj.lc.detrended=None
    obj.lc.detrend_scheme=None
    obj.lc.flare=None
    obj.lc.flare_run=False
    obj.lc.transit=None
    obj.lc.transit_run=False
    obj.flares={"t_start":[],
                    "t_peak":[],
                    "t_stop":[],
                    "i_start":[],
                    "i_stop":[],
                    "amplitude":[],
                    "duration":[],
                    "equi_duration":[],
                    "energy":[]}

def add_flares(obj, N=10):
    """
    Adds flares to the lightcurve.

    Adds N number of flares to the full lightcurve.

    Parameters
    ----------
    obj : InjRec
        Injection recovery objects.
    N : int, optional
        Number of flares to be added, by default 10.

    Attributes
    ----------
    obj.injection : dict
        Dictionary of the injected flare parameters.
    obj.lc.full['flux'] : array
        Flux array with the flares added on top.
    """
    print("Flare addition started.")
    print(f"# Injected flare::{N}")
    obj.injection={'t_peak':[],
                    'i_start': [],
                    'i_stop': [],
                    'ampl': [],
                    'fwhm': [],
                    'ed':[],
                    'energy':[]}

    time=obj.lc.full['time']
    dist_cm=obj.star.dist.to(u.cm)
    cadence=obj.inst.cadence
    model_flux=obj.lc.model['flux']

    arr_time_peak=np.array(sample(list(time), k=N))
    t_peak=arr_time_peak+np.random.uniform(low=-cadence,high=cadence, size=N)
    net_flares_lc=np.zeros(len(time))
    for i in range(len(t_peak)):
        #fwhm is in seconds
        log10_fwhm=np.random.uniform(low=10**1.0,high=10**3.5)
        #ampl is in counts/s
        log10_ampl=np.random.uniform(low=10**1.0,high=10**4.0)

        # fwhm=10**(log10_fwhm)/(24*3600)
        # ampl=10**log10_ampl

        fwhm=log10_fwhm/(24*3600)
        ampl=log10_ampl
        flares_lc=aflare1(time, t_peak[i], fwhm, ampl)

        net_flares_lc=net_flares_lc+flares_lc

        try:
            mask=np.where(flares_lc>1)[0]
            start=mask[0]
            stop=mask[-1]
        except:
            print("Flare too small.")
            arg=np.where(time == arr_time_peak[i])[0][0]
            start=arg-1
            stop=arg+1

        #ED calculation of the flare
        y=flares_lc/model_flux

        in_flare_time=time[start: stop+1]
        in_flare_flux_norm=y[start: stop+1]
        ed=simpson(in_flare_flux_norm, x=in_flare_time)*24*3600
        obj.injection['ed'].append(ed)

        #energy calculation of the flare
        e_count=simpson(flares_lc[start:stop+1], x=time[start:stop+1])*24*3600

        lambda_mean=7452.64*u.angstrom

        h=const.h
        c=const.c

        h_cgs = h.to(u.erg * u.s)
        c_cgs = c.to(u.cm / u.s)
        energy_per_electron = h_cgs * c_cgs / lambda_mean.to(u.cm)

        # Calculate total electron energy and energy
        tot_electron_energy = energy_per_electron * e_count
        # calculates the energy, 86.6 cm2 is aperture area.
        energy = 4 * np.pi * dist_cm**2 * tot_electron_energy/86.6
        
        obj.injection['t_peak'].append(t_peak[i])
        obj.injection['i_start'].append(start)
        obj.injection['i_stop'].append(stop)
        obj.injection['ampl'].append(ampl)
        obj.injection['fwhm'].append(log10_fwhm)
        obj.injection['energy'].append(energy.value)
    
    obj.lc.full['flux']+=net_flares_lc
    print("Flare addition completed.")

def add_flare(obj, t_peak=None, fwhm=200, ampl=1000):
    """
    Adds a flare to the lightcurve.

    Adds a single flare to the full lightcurve.

    Parameters
    ----------
    obj : InjRec
        Injection recovery objects.
    t_peak : float, optional
        Time in mjd of the flare peak, by default None which means random value will be chosen.
    fwhm : float, optional
        Fwhm of the flare, by default 200s.
    ampl : float, optional
        Amplitude of the injected flare, by default 1000 ct/s.

    Attributes
    ----------
    obj.injection : dict
        Dictionary of the injected flare parameters.
    obj.lc.full['flux'] : array
        Flux array with the flares added on top.
    """
    print("Flare addition started.")
    obj.injection={'t_peak':[],
                    'i_start': [],
                    'i_stop': [],
                    'ampl': [],
                    'fwhm': [],
                    'ed':[],
                    'energy':[]}

    time=obj.lc.full['time']
    dist_cm=obj.star.dist.to(u.cm)
    cadence=obj.inst.cadence
    model_flux=obj.lc.model['flux']

    if t_peak==None:
        time_peak=sample(list(time), k=1)[0]
        t_peak=time_peak+np.random.uniform(low=-cadence,high=cadence)

    net_flares_lc=np.zeros(len(time))
    #fwhm is in seconds
    log10_fwhm=np.log10(fwhm)
    #ampl is in counts/s
    log10_ampl=np.log10(ampl)

    fwhm=10**(log10_fwhm)/(24*3600)
    ampl=10**log10_ampl
    flares_lc=aflare1(time, t_peak, fwhm, ampl)

    net_flares_lc=net_flares_lc+flares_lc

    try:
        mask=np.where(flares_lc>1)[0]
        start=mask[0]
        stop=mask[-1]
    except:
        print("Flare too small.")
        arg=np.where(time == time_peak)[0][0]
        pdb.set_trace()
        start=arg-1
        stop=arg+1

    #ED calculation of the flare
    y=flares_lc/model_flux

    in_flare_time=time[start: stop+1]
    in_flare_flux_norm=y[start: stop+1]
    ed=simpson(in_flare_flux_norm, x=in_flare_time)*24*3600
    obj.injection['ed'].append(ed)

    #energy calculation of the flare
    e_count=simpson(flares_lc[start:stop+1], x=time[start:stop+1])*24*3600

    lambda_mean=7452.64*u.angstrom

    h=const.h
    c=const.c

    h_cgs = h.to(u.erg * u.s)
    c_cgs = c.to(u.cm / u.s)
    energy_per_electron = h_cgs * c_cgs / lambda_mean.to(u.cm)

    # Calculate total electron energy and energy
    tot_electron_energy = energy_per_electron * e_count
    # calculates the energy, 86.6 cm2 is aperture area.
    energy = 4 * np.pi * dist_cm**2 * tot_electron_energy/86.6
    
    obj.injection['t_peak'].append(t_peak)
    obj.injection['i_start'].append(start)
    obj.injection['i_stop'].append(stop)
    obj.injection['ampl'].append(10**log10_ampl)
    obj.injection['fwhm'].append(10**log10_fwhm)
    obj.injection['energy'].append(energy.value)
    
    obj.lc.full['flux']+=net_flares_lc
    print("Flare addition completed.")