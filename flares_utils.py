import numpy as np
from FINDflare_dport import FINDflare
from aflare import aflare, aflare1
import pdb
from scipy.integrate import simpson

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
        flare_mask=np.ones(len(obj.lc.segment), dtype=bool)

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
    amplitude of the flare [e/s] : amplitude
    duration of the flare [s] : duration

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.

    Attributes
    ----------
    obj.flares : dict
        {"t_start":, [mjd]
        "t_stop":, [mjd]
        "i_start":,
        "i_stop":,
        "amplitude":, [e/s]
        "duration":, [s]}
    """
    print("Estimating flare parameters.")
    time=obj.lc.detrended['time']
    flux_detrended=obj.lc.detrended['flux']
    t_start=obj.lc.flare['start']
    t_stop=obj.lc.flare['stop']

    for i in range(len(t_start)):
        start, stop=_get_flare_tstart_tstop(time, flux_detrended, t_start[i], t_stop[i])
        obj.flares['t_start'].append(time[start])
        obj.flares['t_stop'].append(time[stop])
        obj.flares["i_start"].append(start)
        obj.flares["i_stop"].append(stop)
        amplitude=np.max(flux_detrended[start:stop+1])
        obj.flares['amplitude'].append(amplitude)
        dur=(time[stop]-time[start])*24*3600
        obj.flares['duration'].append(dur)

def get_ED(obj):
    """
    Calculates Equivalent duration of the flare.

    Calculates Equivalent duration of the flare as stated in section 3.2, Diamond-Lowe et al. 2021.

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
        integral_i_1=simpson(in_flare_flux_i_1, x=in_flare_time_i_1)
        rel_diff=(integral_i_1-integral_i)/integral_i
        if rel_diff>rel_diff_th:
            change_start=True
            t_start=t_start_i_1
        else:
            change_start=False

    return t_start, t_stop


# Adds a single flare in the self.flares.lc
def add_flare1(self,tpeak,fwhm,ampl,q_flags=None,segments=None):
    mask=self.lc.get_mask(q_flags=q_flags, segments=segments)
    t=self.lc.full['time'][mask]
    flux=self.lc.full['flux'][mask]
    flux_err=self.lc.full['flux_err'][mask]
    quality=self.lc.full['quality'][mask]
    flare=aflare1(t,tpeak=tpeak, fwhm=fwhm, ampl=ampl)
    self.flares.lc={"time":t, "flux":flux+flare,
                    "flux_err":flux_err, "quality":quality}