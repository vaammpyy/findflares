import pymc3 as pm
import numpy as np
import aesara_theano_fallback.tensor as tt 
from celerite2.theano import terms, GaussianProcess
import pymc3_ext as pmx
from lc_utils import get_period, get_mask

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
        map_soln = pmx.optimize()
        return map_soln

def GaussianProcess_detrend(obj, segments=None, mask_flare=False):
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
        If true then it will mask the flares when training the model, by default False. When true it assumes obj.flares is defined.

    Attributes
    ----------
    obj.lc.detrended : dict
        Detrended lightcurve dictionary.
    obj.lc.model : ndarray
        Model used for detrending lightcurve.
    """
    q_flags=[0]

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

        # Masking the flares
        if mask_flare:
            flare_mask=obj.lc.flare['mask']
            comb_model_mask = flare_mask & model_mask
        else:
            comb_model_mask = model_mask

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