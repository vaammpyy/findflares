import numpy as np
import exoplanet as xo
import matplotlib.pyplot as plt

def segment_lightcurve(obj, period, factor=1, min_segment_len=2):
    """
    Segments the lightcurve.

    Segments the lightcurve over the data gaps that are larger than <factor>*time period of the stellar rotation.
    No segment shorter than <min_segment_len> are taken as the data is not sufficient for a decent GP regression.

    Parameters
    ----------
    obj : TESSLC
        TESSLC object.
    period (days) : float
        Time period of the stellar rotation.
    factor : float, optional
        How much times the <time_period> is the allowed gap, by default 1. Chosen by the memory of the GP cavariance.
    min_segment_len (hours) : float, optional
        Minimum length of the allowed segment, by default 2.
    
    Attributes
    ----------
    obj.lc.segment : ndarray
        Segments masks starting from 1 upto total number of segments, 0 everywhere else. All the data points with segment mask greater
        than 0 means that the data point is a part of the segment.
    """
    t_gap= np.where(np.diff(obj.lc.full['time'])>obj.inst.cadence+obj.inst.cadence_err)[0]

    segment=[]

    for i in range(len(t_gap)):
        t_i=obj.lc.full['time'][t_gap[i]]
        t_i_1=obj.lc.full['time'][t_gap[i]+1]
        diff_t=t_i_1-t_i
        if diff_t>factor*period: # Taking the gap to be greater than factor*period for segmentation.
            segment.append(t_gap[i]) 

    segment=np.insert(segment,0,-1)
    segment=np.append(segment,len(obj.lc.full['time'])-1)

    min_segment_length=min_segment_len/24 # defining the threshold segment length of 2hr i.e. 1/12 [d].

    segment_mask=np.zeros(len(obj.lc.full['time']))

    segment_number=1
    for i in range(1,len(segment)):
        if obj.lc.full['time'][segment[i]]-obj.lc.full['time'][segment[i-1]+1]>min_segment_length:
            segment_mask[segment[i-1]+1:segment[i]+1]=segment_number
            segment_number+=1
    
    obj.lc.segment=segment_mask

def get_period(obj, mask):
    """
    Evaluates dominant period using the GLS periodogram.

    Evaluates the Generalized Lombscargle Periodogram (GLS) for the data and returns the time period of the dominant period.
    Which coresponds to the stellar rotation.

    Parameters
    ----------
    obj : TESSLC
        TESSLC lightcurve object.
    mask : ndarray
        Mask to be applied to the data for evaluating the lomb_scargle periodogram.

    Returns
    -------
    period : float
        Time period of the dominant peak.
    """
    data=obj.lc.full
    time=data['time'][mask]
    flux=data['flux'][mask]
    flux_err=data['flux_err'][mask]
    result=xo.lomb_scargle_estimator(time, flux, yerr=flux_err, max_peaks=1, min_period=0.01, max_period=200.0, samples_per_peak=50)
    return result['peaks'][0]['period']

def get_mask(obj, q_flags=None, segments=None):
    """
    Positive mask for the select quality and segment.

    Parameters
    ----------
    obj : TESSLC
        TESSLC lightcurve object.
    q_flags : list, optional
        Quality flags to be considered, by default None.
    segments : list
        Segments to be considered, by default None.

    Returns
    -------
    mask : ndarray
        Positive mask with the given q_flags and segments.

    Notes
    -----
    If no q_flags and segments is parsed, all the segments and q_flags will be considered for the mask.
    """
    if q_flags is None:
        q_mask=np.ones_like(obj.lc.full['time'], dtype=bool)
    else:
        q_mask=np.isin(obj.lc.full['quality'],np.asarray(q_flags))

    if segments is None:
        seg_mask=np.ones_like(obj.lc.full['time'], dtype=bool)
    else:
        seg_mask=np.isin(obj.lc.segment,np.asarray(segments))

    mask= q_mask & seg_mask
    return(mask)

def clean_lightcurve(obj):
    """
    Cleans lightcurve.

    Removes nans and data points with q_flag other than [0,512] from all the columns in the lightcurve data.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    
    Attributes
    ----------
    obj.lc.full : dict
        Cleaned data points, all values finite.
    obj.lc.segment : ndarray
        Cleaned segment mask, segment mask corresponding to all the finite data points.
    
    Notes
    -----
    Run segment_lightcurve before running this as this function assumes that the obj.lc.segment is defined before.
    """
    data=obj.lc.full
    time=data['time']
    flux=data['flux']
    flux_err=data['flux_err']
    quality=data['quality']
    finite_time_mask=np.isfinite(time)
    finite_flux_mask=np.isfinite(flux)
    finite_flux_err_mask=np.isfinite(flux_err)
    finite_quality_mask=np.isfinite(quality)

    finite_mask=finite_flux_mask & finite_time_mask & finite_flux_err_mask & finite_quality_mask

    quality_mask=get_mask(obj, q_flags=[0,512]) # getting the quality_mask from the

    combined_mask= finite_mask & quality_mask

    obj.lc.full = {'time': time[combined_mask],
                'flux': flux[combined_mask],
                'flux_err': flux_err[combined_mask],
                'quality': quality[combined_mask]} 

    obj.lc.segment=  obj.lc.segment[combined_mask]

def plot_lightcurve(obj, mode=None, q_flags=None, segments=None):
    """
    Plots lightcurve.

    Parameter
    ---------
    obj : TESSLC
        TESS lightcurve object.
    mode : str, optional
        Plotting mode, by default None.
        None : Full lightcurve
        'model_overlay' : Full lightcurve with model overlaid.
        'detrended' : Detrended lightcurve.
    q_flags : list, optional
        Quality flags of the data to be plotted, by default None.
    segments : list, optional
        Segment of the data to be plotted, by default None.
    """
    mask=get_mask(obj, q_flags=q_flags, segments=segments)
    fig=plt.figure(figsize=(20,10), facecolor='white')
    if mode is None:
        plt.scatter(obj.lc.full['time'][mask],obj.lc.full['flux'][mask], s=0.01, color='k', label=f"TIC {obj.TIC}")
    if mode == 'model_overlay':
        plt.scatter(obj.lc.full['time'][mask],obj.lc.full['flux'][mask], s=0.01, color='k', label=f"TIC {obj.TIC}")
        plt.plot(obj.lc.model['time'][mask],obj.lc.model['flux'][mask], 'r')
    if mode == 'detrended':
        plt.scatter(obj.lc.detrended['time'][mask],obj.lc.detrended['flux'][mask], s=0.01, color='k', label=f"TIC {obj.TIC}")
    plt.xlabel("Time [mjd - 2,457,000]", fontsize=14)
    plt.ylabel("Flux [e/s]", fontsize=14)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()