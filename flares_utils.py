import numpy as np
from FINDflare_dport import FINDflare
from aflare import aflare, aflare1

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

def _include_tail(data, start_indices, stop_indices, sig_lvl):
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
    
    # Determine the threshold
    threshold = mean + sig_lvl * std_dev
    
    # Initialize new start and stop indices
    new_start_indices = []
    new_stop_indices = []

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

def find_flare(obj):
    """
    Finds flares in the lightcurve.

    Finds flares in the lightcurve object using the modified critrion described in Chang et al. 2015 and Davenport 2016.
    Start and stop times the flares are also extended by merging the close flares and tracking the tail as described in
    Davenport 2016 and Ilin and Poppenhaeger 2022.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.

    Attributes
    ----------
    obj.lc.flare : dict
        Flare dictionary, {'start', 'stop', 'mask'}, note that the flare mask is negative i.e 0 means flare point.
    """
    data=obj.lc.detrended
    flux=data['flux']
    flux_err=data['flux_err']
    start, stop=FINDflare(flux, flux_err, N1=3, N2=1, N3=3, avg_std=True, std_window=5)
    start, stop= _include_tail(flux, start, stop, sig_lvl=1)
    start, stop=_merge_flares(start, stop, close_th=9)

    #making a negative flare mask
    flare_mask=np.ones(len(obj.lc.segment), dtype=bool)
    for i in range(len(start)):
        flare_mask[start[i]:stop[i]+1]=0

    obj.lc.flare={'start': start,
                'stop': stop,
                'mask': flare_mask}

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