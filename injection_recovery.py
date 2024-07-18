import copy
import numpy as np
from imports import *

def log_injection_recovery(obj, rec_index=None, inj_index=None, flag="-1.-1.-1"):
    """
    Logs injection recovery flares.

    Logs injection recovery flares with detection flags.
    
    Parameters
    ----------
    obj : InjRec
        Injection recovery class.
    rec_index : int, optional
        Recovered flare index, by default None.
    inj_index : int, optional
        Injected flare index, by default None.
    flag : str, optional
        Flag result of injection-recovery test, by default "-1.-1.-1"

    Attributes
    ----------
    obj.injrec : list
        List of injection recovery dictionaries.
    """
    inj_dict=obj.injection
    rec_dict=obj.flares

    if rec_index!=None and inj_index!=None:
        inj_rec_dict={
            "injected":{"t_peak":inj_dict['t_peak'][inj_index],
                        "i_start":inj_dict['i_start'][inj_index],
                        "t_stop":inj_dict['i_stop'][inj_index],
                        "ampl":inj_dict['ampl'][inj_index],
                        "fwhm":inj_dict['fwhm'][inj_index],
                        "ed":inj_dict['ed'][inj_index],
                        "energy":inj_dict['energy'][inj_index]},
            "recovered":{"t_start":rec_dict['t_start'][rec_index],
                        "t_stop":rec_dict["t_stop"][rec_index],
                        "i_start":rec_dict['i_start'][rec_index],
                        "i_stop":rec_dict["i_stop"][rec_index],
                        "amplitude":rec_dict['amplitude'][rec_index],
                        "duration":rec_dict["amplitude"][rec_index],
                        "equi_duration":rec_dict['equi_duration'][rec_index],
                        "energy":rec_dict['energy'][rec_index]},
            "flag":flag
        }
    elif inj_index==None:
        inj_rec_dict={
            "injected":{},
            "recovered":{"t_start":rec_dict['t_start'][rec_index],
                        "t_stop":rec_dict["t_stop"][rec_index],
                        "i_start":rec_dict['i_start'][rec_index],
                        "i_stop":rec_dict["i_stop"][rec_index],
                        "amplitude":rec_dict['amplitude'][rec_index],
                        "duration":rec_dict["amplitude"][rec_index],
                        "equi_duration":rec_dict['equi_duration'][rec_index],
                        "energy":rec_dict['energy'][rec_index]},
            "flag":flag
        }
    elif rec_index==None:
        inj_rec_dict={
            "injected":{"t_peak":inj_dict['t_peak'][inj_index],
                        "i_start":inj_dict['i_start'][inj_index],
                        "t_stop":inj_dict['i_stop'][inj_index],
                        "ampl":inj_dict['ampl'][inj_index],
                        "fwhm":inj_dict['fwhm'][inj_index],
                        "ed":inj_dict['ed'][inj_index],
                        "energy":inj_dict['energy'][inj_index]},
            "recovered":{},
            "flag":flag
        }
    obj.injrec.append(inj_rec_dict)

def recover_flares(obj, run):
    """
    Checks recovery of injected flares.

    Checks recovery of the injected flares. Recovery criterion of the injected flare,
    If the injected flare lies within the start and stop of the detected flare then it's
    deemed recovered.

    Recovery flag structure
    -----------------------
    flag=<run>.<flare_event>.<sub_flare_#>

    <run> : {1,2,3,...}
    <flare_event>
        if injected flare is detected : {1,2,3,...}
        if injected flare is not detected : 0
        if false positive detection : -
    <sub_flare_#>
        if injected flare is detected : {1,2,3,...} [Note: sub_flare_# 1 corresponds to the detected flare whose peak is closest to the injected flare]
        if injected flare is not detected : 0
        if false positive detection : 0

    Interesting cases
    -----------------
    Case 1: Overlap detection (Injected flare: 2, Recovered flare: 1)
            If two injected flares are detected as one, then the flare detected peak closest
            to the injected one is tagged recovered, other "sub-flares" are considered not recovered
            as the detection was not done individually by definition.
            Flag example,
            1) Injected flare that is being detected: run.flare_event.1
            2) Injected flare that is being detected as sub-flare: run.flare_event.2
    Case 2: One detected and other not detected (Injected flare: 2, Recovered flare: 1)
            Flag example,
            1) Injected flare that is being detected: run.flare_event.1
            2) Injected flare that is not being detected: run.0.0
    Case 3: One flare detected, one flase positive (Injected flare: 2, Recovered flare: 1, False positive: 1)
            Flag example,
            1) Injected flare that is being detected: run.flare_event.1
            2) Injected flare that is not being detected: run.0.0
            2) Injected flare that is being detected as flase positive: run.-.0
    
    Parameters
    ----------
    obj : InjRec
        Injection recovery class.
    run : int
        # run of the injection recovery test.
    
    Attributes
    ----------
    obj.injrec : array
        Array of injection recovery dictionaries.
    """
    print("Flare recovery started.")
    inj_dict=copy.copy(obj.injection)
    rec_dict=copy.copy(obj.flares)

    n_detected_flare=len(obj.flares['t_start'])

    t_peak_Inj=inj_dict['t_peak']

    detected_injections=[]

    for i in range(n_detected_flare):
        t_peak_rec=rec_dict['t_peak'][i]
        t_start=rec_dict['t_start'][i]
        t_stop=rec_dict['t_stop'][i]
        mask=(t_peak_Inj>=t_start) & (t_peak_Inj<=t_stop)
        index_array = np.nonzero(mask)[0]
        t_peak_diff=np.array(t_peak_Inj)-t_peak_rec
        sorted_order=np.argsort(t_peak_diff[index_array])
        sorted_index=index_array[sorted_order]
        flare_number=1
        if sorted_index.size==0:
            flag=f"{run:02d}.-.{0}"
            log_injection_recovery(obj,rec_index=i,flag=flag)
        else:
            for inj_index in sorted_index:
                flag=f"{run:02d}.{i+1}.{flare_number}"
                flare_number+=1
                detected_injections.append(inj_index)
                log_injection_recovery(obj,rec_index=i, inj_index=inj_index, flag=flag)

    n_injected_flares=len(inj_dict['t_peak'])

    for k in range(n_injected_flares):
        if k in detected_injections:
            continue
        flag=f"{run:02d}.{0}.{0}"
        log_injection_recovery(obj,inj_index=k,flag=flag)
    print("Flare recovery completed.")