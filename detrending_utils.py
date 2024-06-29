import numpy as np
import pymc as pm
import pymc_ext as pmx
import pytensor.tensor as tt
from celerite2.pymc import terms, GaussianProcess
from lc_utils import get_period, get_mask
from scipy.signal import savgol_filter
from gls import *
from flares_utils import find_flare
import matplotlib.pyplot as plt
import pdb

def RotationTerm_model(obj, model_mask, eval_mask):
    """
    Rotation term kernel for gaussian process modelling.

    Rotation term kernel for gaussian process modelling of stellar rotation, section 5.2.1 in Medina et al. 2020.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    model_mask : ndarray
        Mask for training the model, includes data points with q_flag = [0].
    eval_mask : ndarray
        Mask to evaluate the model, for detrending all the data points with q_flags i.e [0,512].

    Returns
    -------
    map_soln : obj
        Holds the evaluated model at different time stamps and other information.
    """
    x=obj.lc.full['time'][model_mask]
    y=obj.lc.full['flux'][model_mask]
    yerr=obj.lc.full['flux_err'][model_mask]

    x_eval=obj.lc.full['time'][eval_mask]

    period_peak=get_period(obj, model_mask)

    with pm.Model() as model:
        # The mean flux of the time series
        mean = pm.Normal("mean", mu=np.mean(y), sigma=10)

        # A jitter term describing excess white noise
        log_jitter = pm.Uniform("log_jitter", lower=-3, upper=2)
        #log_jitter = pm.Normal("log_jitter", mu=np.log(np.var(y)), sigma=10)

        # The parameters of the RotationTerm kernel
        log_sigma_rot=pm.Uniform("log_sigma_rot", lower=-1, upper=2)

        log_period = pm.Normal("log_period", mu=np.log(period_peak), sigma=0.01) # sigma 0.01
        period = pm.Deterministic("period", tt.exp(log_period))
        Q0 = pm.Normal("Q0", mu=5, sigma=2)
        log_dQ = pm.Normal("log_dQ", mu=0, sigma=1) # mu = 10.0
        f = pm.Uniform("f", lower=0.8, upper=1)

        # Set up the Gaussian Process model
        kernel = terms.RotationTerm(
            sigma=tt.exp(log_sigma_rot),
            period=period,
            Q0=Q0,
            dQ=tt.exp(log_dQ),
            f=f,
        )
        gp = GaussianProcess(
            kernel,
            t=x,
            diag=yerr**2 + tt.exp(2 * log_jitter),
            mean=mean,
            quiet=True,
        )

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("gp", observed=y)

        # Compute the mean model prediction for plotting purposes
        # Evaluating the values from the GP at all the raw LC time stamps
        pm.Deterministic("pred", gp.predict(y, t=x_eval))

        # Optimize to find the maximum a posterior parameters
        map_soln = pmx.optimize(progressbar=False)
        return map_soln

def GaussianProcess_detrend(obj, segments=None, mask_flare=False, mask_transit=False, mask_outlier=False, iter=False):
    """
    Detrends lightcurve data using gaussian process regression.

    Detrends lightcurve data using gaussian process regression with the Rotation kernel.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    segments : list, optional
        Segments to detrend, by default None. If not parsed then detrends all the segments.
    mask_flare : bool, optional
        If true then it will mask the flares when training the model, by default False.
        When true it assumes obj.lc.flares is defined.
    mask_transit : bool, optional
        If true then it will mask the transit when training the model, by default False.
        When true it assumes obj.lc.transit is defined.
    mask_outlier : bool, optional
        When True outliers (>3*sig) will be masked out, by default False.
    iter : bool, optional
        When True GP runs iteratively for each segment, by default False.

    Attributes
    ----------
    obj.lc.detrended : dict
        Detrended lightcurve dictionary.
    obj.lc.model : ndarray
        Model used for detrending lightcurve.
    """
    print("GP DETREND STARTED")
    q_flags=[0]

    # if iter==None or iter==1:
    lc_detrended={'time':np.array([]),
                    "flux":np.array([]),
                    "flux_err":np.array([]),
                    "quality":np.array([])}

    model_lc={'time':np.array([]),
                "flux":np.array([])}

    if segments is None:
        segments=np.unique(obj.lc.segment)
    
    print(f"GP modelling segments: {segments}")
    for seg in segments:
        print()
        print("------------------------------------")
        print(f"Segment: {seg}, started.")
        model_mask=get_mask(obj, q_flags=q_flags, segments=[seg])

        if iter:
            rotation=True
            i=1
            mask_flare=False
            mask_transit=False
            mask_outlier=True
            while rotation:
                print(f"Iter::{i}")
                if mask_flare:
                    flare_mask=obj.lc.flare['mask']
                    if flare_mask.size==0:
                        comb_model_mask = model_mask
                    else:
                        comb_model_mask = flare_mask & model_mask
                else:
                    comb_model_mask = model_mask

                if mask_transit:
                    transit_mask=obj.lc.transit['mask']
                    comb_model_mask = transit_mask & comb_model_mask
                
                if mask_outlier:
                    flux=obj.lc.full['flux']
                    mean=np.mean(flux)
                    std=np.std(flux)
                    flux_dev=abs(flux-mean)
                    outlier_mask=flux_dev<3*std
                    comb_model_mask = outlier_mask & comb_model_mask

                eval_mask=get_mask(obj, segments=[seg])
                map_soln=RotationTerm_model(obj, model_mask=comb_model_mask, eval_mask=eval_mask)

                time=obj.lc.full['time'][eval_mask]
                flux_detrended=obj.lc.full['flux'][eval_mask]-map_soln['pred']
                flux_err=obj.lc.full['flux_err'][eval_mask]
                gls=Gls(((time, flux_detrended, flux_err)), fend=10, fbeg=1/14)
                fap=gls.FAP()
                mask_flare=True
                mask_transit=True
                mask_outlier=True
                if fap<0.001:
                    print(f"FAP::{fap}, rotation found.")
                    find_flare(obj, find_transit=mask_transit)
                    if i>3:
                        rotation= False
                    else:
                        rotation= True
                else:
                    print(f"FAP::{fap}, rotation not found.")
                    rotation= False

                # fig=plt.figure(figsize=(20,10))
                # plt.scatter(time, obj.lc.full['flux'][eval_mask], s=0.01, color='k')
                # plt.plot(time, map_soln['pred'], color='magenta')
                # plt.savefig(f"./{i}.png")
                i+=1

            lc_detrended['time']= np.append(lc_detrended['time'],obj.lc.full['time'][eval_mask])
            lc_detrended['flux']= np.append(lc_detrended['flux'],obj.lc.full['flux'][eval_mask]-map_soln['pred'])
            lc_detrended['flux_err']= np.append(lc_detrended['flux_err'],obj.lc.full['flux_err'][eval_mask])
            lc_detrended['quality']= np.append(lc_detrended['quality'],obj.lc.full['quality'][eval_mask])

            model_lc['time']= np.append(model_lc['time'],obj.lc.full['time'][eval_mask])
            model_lc['flux']= np.append(model_lc['flux'],map_soln['pred'])

        else:
            # Masking the flares when flares are detected.
            if mask_flare and len(obj.lc.flare['mask'])>0:
                flare_mask=obj.lc.flare['mask']
                comb_model_mask = flare_mask & model_mask
            else:
                comb_model_mask = model_mask

            if mask_transit:
                transit_mask=obj.lc.transit['mask']
                comb_model_mask = transit_mask & comb_model_mask
            
            if mask_outlier:
                flux=obj.lc.full['flux']
                mean=np.mean(flux)
                std=np.std(flux)
                flux_dev=abs(flux-mean)
                outlier_mask=flux_dev<3*std
                comb_model_mask = outlier_mask & comb_model_mask

            eval_mask=get_mask(obj, segments=[seg])
            map_soln=RotationTerm_model(obj, model_mask=comb_model_mask, eval_mask=eval_mask)

            lc_detrended['time']= np.append(lc_detrended['time'],obj.lc.full['time'][eval_mask])
            lc_detrended['flux']= np.append(lc_detrended['flux'],obj.lc.full['flux'][eval_mask]-map_soln['pred'])
            lc_detrended['flux_err']= np.append(lc_detrended['flux_err'],obj.lc.full['flux_err'][eval_mask])
            lc_detrended['quality']= np.append(lc_detrended['quality'],obj.lc.full['quality'][eval_mask])

            model_lc['time']= np.append(model_lc['time'],obj.lc.full['time'][eval_mask])
            model_lc['flux']= np.append(model_lc['flux'],map_soln['pred'])
        print(f"Segment: {seg}, completed.")

    obj.lc.detrended=lc_detrended
    obj.lc.model=model_lc

def Median_detrend(obj, segments=None, window_length=12, mask_flare=False, mask_transit=False):
    """
    Median filter detrending.

    Median filter detrending using first order savitzky-golay over a 12hr window.

    Parameters
    ----------
    obj : TESSLC
        TESS lightcurve object.
    segments : list, optional
        Segments to detrend, by default 'None' i.e detrending over all segments.
    window_length (hrs) : int, optional
        Window length for median filter, by default 12hr.
    mask_flares: bool, optional
        If True flares will be masked out, by default False.
    mask_transit: bool, optional
        If True transits will be masked out, by default False.
    
    Attributes
    ----------
    obj.lc.detrended : dict
        Detrended lightcurve data.
    obj.lc.model : None
        None as there is no model.
    
    Notes
    -----
    This module should only be run when no rotation is found in the lightcurve.
    The median filter window is static making the effective smoothing longer, because of the data gaps while cleaning.
    """
    print("MEDIAN DETREND STARTED")
    # lc_detrended={'time':np.array([]),
    #                 "flux":np.array([]),
    #                 "flux_err":np.array([]),
    #                 "quality":np.array([])}

    # if segments is None:
    #     segments=np.unique(obj.lc.segment)
    
    # for seg in segments:
    #     print(f"Segment: {seg}, started.")
    #     model_mask=get_mask(obj, segments=[seg])

    #     if mask_flare and len(obj.lc.flare['mask'])>0:
    #         flare_mask=obj.lc.flare['mask']
    #         comb_model_mask = flare_mask & model_mask
    #     else:
    #         comb_model_mask = model_mask

    #     if mask_transit:
    #         transit_mask=obj.lc.transit['mask']
    #         comb_model_mask = transit_mask & comb_model_mask

    #     time=obj.lc.full['time'][comb_model_mask]
    #     flux=obj.lc.full['flux'][comb_model_mask]
    #     flux_err=obj.lc.full['flux_err'][comb_model_mask]
    #     quality=obj.lc.full['quality'][comb_model_mask]
        
    #     cadence=obj.inst.cadence*3600*24
    #     window_length_dp=int(window_length*3600/cadence)

    #     #making window length odd
    #     if window_length_dp%2==0:
    #         window_length_dp+=1

    #     sav_model = savgol_filter(flux, window_length_dp, 1) - 1
    #     flux_detrended = flux - sav_model

    #     lc_detrended['time']= np.append(lc_detrended['time'],time)
    #     lc_detrended['flux']= np.append(lc_detrended['flux'],flux_detrended)
    #     lc_detrended['flux_err']= np.append(lc_detrended['flux_err'],flux_err)
    #     lc_detrended['quality']= np.append(lc_detrended['quality'],quality)

    #     print(f"Segment: {seg}, completed.")
    # if mask_flare and len(obj.lc.flare['mask'])>0:
    #     flare_mask=obj.lc.flare['mask']
    #     comb_model_mask = flare_mask & model_mask
    # else:
    #     comb_model_mask = model_mask

    # if mask_transit:
    #     transit_mask=obj.lc.transit['mask']
    #     comb_model_mask = transit_mask & comb_model_mask

    time=obj.lc.full['time']
    flux=obj.lc.full['flux']
    flux_err=obj.lc.full['flux_err']
    quality=obj.lc.full['quality']
    
    cadence=obj.inst.cadence*3600*24
    window_length_dp=int(window_length*3600/cadence)

    #making window length odd
    if window_length_dp%2==0:
        window_length_dp+=1

    sav_model = savgol_filter(flux, window_length_dp, 1) - 1
    flux_detrended = flux - sav_model

    lc_detrended={'time':time,
                    "flux":flux_detrended,
                    "flux_err":flux_err,
                    "quality":quality}

    obj.lc.detrended=lc_detrended
    obj.lc.model=None