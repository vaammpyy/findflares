import copy
import numpy as np
from imports import *
import fnmatch

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

    <run> : {01,02,03,...}
    <flare_event>
        if injected flare is detected : {01,02,03,...}
        if injected flare is not detected : 00
        if false positive detection : --
    <sub_flare_#>
        if injected flare is detected : {01,02,03,...} [Note: sub_flare_# 1 corresponds to the detected flare whose peak is closest to the injected flare]
        if injected flare is not detected : 00
        if false positive detection : 00

    Interesting cases
    -----------------
    Case 1: Overlap detection (Injected flare: 2, Recovered flare: 1)
            If two injected flares are detected as one, then the flare detected peak closest
            to the injected one is tagged recovered, other "sub-flares" are considered not recovered
            as the detection was not done individually by definition.
            Flag example,
            1) Injected flare that is being detected: run.flare_event.01
            2) Injected flare that is being detected as sub-flare: run.flare_event.02
    Case 2: One detected and other not detected (Injected flare: 2, Recovered flare: 1)
            Flag example,
            1) Injected flare that is being detected: run.flare_event.01
            2) Injected flare that is not being detected: run.00.00
    Case 3: One flare detected, one flase positive (Injected flare: 2, Recovered flare: 1, False positive: 1)
            Flag example,
            1) Injected flare that is being detected: run.flare_event.1
            2) Injected flare that is not being detected: run.00.00
            2) Injected flare that is being detected as flase positive: run.--.00
    
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
            flag=f"{run:02d}.--.{0:02d}"
            log_injection_recovery(obj,rec_index=i,flag=flag)
        else:
            for inj_index in sorted_index:
                flag=f"{run:02d}.{i+1:02d}.{flare_number:02d}"
                flare_number+=1
                detected_injections.append(inj_index)
                log_injection_recovery(obj,rec_index=i, inj_index=inj_index, flag=flag)

    n_injected_flares=len(inj_dict['t_peak'])

    for k in range(n_injected_flares):
        if k in detected_injections:
            continue
        flag=f"{run:02d}.{0:02d}.{0:02d}"
        log_injection_recovery(obj,inj_index=k,flag=flag)
    print("Flare recovery completed.")

def get_ir_mask(flags, mode=None):
    flags=np.array(flags)

    mask_rec=np.zeros(len(flags), dtype=bool)
    mask_fp=np.zeros(len(flags), dtype=bool)
    mask_inj=np.zeros(len(flags), dtype=bool)

    if 'rec' in mode:
        pattern='??.??.01'
        mask_rec=np.array([fnmatch.fnmatch(flag, pattern) for flag in flags], dtype=bool)

    if 'fp' in mode:
        pattern='??.--.00'
        mask_fp=np.array([fnmatch.fnmatch(flag, pattern) for flag in flags], dtype=bool)

    if 'inj' in mode:
        pattern='??.?[!--].??'
        mask_inj=np.array([fnmatch.fnmatch(flag, pattern) for flag in flags], dtype=bool)

    mask=np.bitwise_or.reduce([mask_rec, mask_inj, mask_fp])

    return mask

def plot_ir_results(obj, mode=None, save_fig=False):
       injrec=obj.injrec
       flags=[injrec[i]['flag'] for i in range(len(injrec))]

       if mode=='rec_frac':
              fName=f"{mode}_{obj.inst.sector}_{int(obj.inst.cadence*24*3600)}.png"
              mask_rec=get_ir_mask(flags=flags, mode=['rec'])
              mask_inj=get_ir_mask(flags=flags, mode=['inj'])

              recovered=np.array(injrec)[mask_rec]
              injected=np.array(injrec)[mask_inj]

              recovered_fwhm=np.log10(np.array([recovered[i]["injected"]['fwhm'] for i in range(len(recovered))]))
              recovered_ampl=np.log10(np.array([recovered[i]["injected"]['ampl'] for i in range(len(recovered))]))

              injected_fwhm=np.log10(np.array([injected[i]["injected"]['fwhm'] for i in range(len(injected))]))
              injected_ampl=np.log10(np.array([injected[i]["injected"]['ampl'] for i in range(len(injected))]))

              recovered_hist2d=np.histogram2d(recovered_fwhm, recovered_ampl, bins=[np.linspace(1,3.5,10), np.linspace(1,4,10)])
              injected_hist2d=np.histogram2d(injected_fwhm, injected_ampl, bins=[np.linspace(1,3.5,10), np.linspace(1,4,10)])

              rec_frac=recovered_hist2d[0]/injected_hist2d[0]

              xedges=injected_hist2d[1]
              yedges=injected_hist2d[2]

              fig=plt.figure(figsize=(6,6))
              plt.imshow(rec_frac.T,origin='lower', aspect='equal',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='coolwarm')
              plt.xlabel("log10(FWHM)", fontsize=14)
              plt.ylabel("log10(Ampl)", fontsize=14)
              plt.title("Recovery Fraction", fontsize=16)
              plt.colorbar()

       if mode=='erg_comp':
              fName=f"{mode}_{obj.inst.sector}_{int(obj.inst.cadence*24*3600)}.png"
              mask_rec=get_ir_mask(flags=flags, mode=['rec'])
              recovered=np.array(injrec)[mask_rec]

              recovered_energy=np.log10(np.array([recovered[i]["recovered"]['energy'] for i in range(len(recovered))]))
              injected_energy=np.log10(np.array([recovered[i]["injected"]['energy'] for i in range(len(recovered))]))
              injected_fwhm=np.log10(np.array([recovered[i]["injected"]['fwhm'] for i in range(len(recovered))]))
              injected_ampl=np.log10(np.array([recovered[i]["injected"]['ampl'] for i in range(len(recovered))]))

              erg_frac=recovered_energy/injected_energy
              fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
              # plt.scatter(injected_energy, erg_frac, c='blue', edgecolor='black', alpha=0.6)
              ax1.scatter(injected_energy, erg_frac, c='blue', edgecolor='black', alpha=0.4)
              ax1.axhline(y=1, color='r', linestyle='--')
              ax1.set_ylabel("Recovered energy fraction", fontsize=14)
              ax1.set_xlabel("Log Injected Energy (ergs)", fontsize=14)

              ax2.scatter(injected_fwhm, erg_frac, c='blue', edgecolor='black', alpha=0.4)
              ax2.axhline(y=1, color='r', linestyle='--')
              # ax2.set_ylabel("Recovered energy fraction")
              ax2.set_xlabel("Log FWHM (s)", fontsize=14)

              ax3.scatter(injected_ampl, erg_frac, c='blue', edgecolor='black', alpha=0.4)
              ax3.axhline(y=1, color='r', linestyle='--')
              # ax3.set_ylabel("Recovered energy fraction")
              ax3.set_xlabel("Log Amplitude (ct/s)", fontsize=14)
              plt.subplots_adjust(wspace=0.05, hspace=0)
       
       if mode=='fp':
              fName=f"{mode}_{obj.inst.sector}_{int(obj.inst.cadence*24*3600)}.png"
              mask_rec=get_ir_mask(flags=flags, mode=['fp', 'rec'])
              mask_fp=get_ir_mask(flags=flags, mode=['fp'])

              false_positives=np.array(injrec)[mask_fp]
              recovered=np.array(injrec)[mask_rec]

              flase_positives_energy=np.log10(np.array([false_positives[i]["recovered"]['energy'] for i in range(len(false_positives))]))
              recovered_energy=np.log10(np.array([recovered[i]["recovered"]['energy'] for i in range(len(recovered))]))

              bin_edges=np.linspace(28,34,12)

              fp_hist=np.histogram(flase_positives_energy, bins=bin_edges)
              recovered_hist=np.histogram(recovered_energy, bins=bin_edges)

              fp_frac=fp_hist[0]/recovered_hist[0]

              fig=plt.figure(figsize=(6,6))
              plt.bar(bin_edges[:-1], fp_frac, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
              plt.xlabel("log10(Energy) (ergs)", fontsize=14)
              plt.ylabel("Fractional False Positive", fontsize=14)

       if mode=='rec_frac_erg':
              fName=f"{mode}_{obj.inst.sector}_{int(obj.inst.cadence*24*3600)}.png"
              mask_rec=get_ir_mask(flags=flags, mode=['rec'])
              mask_inj=get_ir_mask(flags=flags, mode=['inj'])

              injected=np.array(injrec)[mask_inj]
              recovered=np.array(injrec)[mask_rec]

              injected_energy=np.log10(np.array([injected[i]["injected"]['energy'] for i in range(len(injected))]))
              recovered_energy=np.log10(np.array([recovered[i]["injected"]['energy'] for i in range(len(recovered))]))

              bin_edges=np.linspace(28,34,20)

              injected_hist=np.histogram(injected_energy, bins=bin_edges)
              recovered_hist=np.histogram(recovered_energy, bins=bin_edges)

              fp_frac=recovered_hist[0]/injected_hist[0]

              fig=plt.figure(figsize=(6,6))
              plt.bar(bin_edges[:-1], fp_frac, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
              plt.xlabel("log10(Energy) (ergs)", fontsize=14)
              plt.ylabel("Fractional Detection", fontsize=14)

       if save_fig:
              plt.savefig(f"{obj.dir}/{fName}", dpi=100)
              print(f"Plot saved.")
              print(f"PATH::{obj.dir}/{fName}.")
       else:
              plt.show()