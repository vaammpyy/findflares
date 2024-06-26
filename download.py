import lightkurve as lk
import numpy as np

def search_lightcurve(tic, cadence=20, mission="TESS", author="SPOC", ret_list=False):
    """
    Searches for lightcurve for a given tic and returns list tuples (tic, sector).

    Parameters
    ----------
    tic : int
        TIC number of the object.
    cadence : list, optional
        List of observation cadence, by default 20.
    mission : str, optional
        Observation mission, by default "TESS".
    author : str, optional
        Author of the observation, by default "SPOC".
    ret_list : bool, optional
        If true returns list of (tic, sector), by default False.
    
    Returns
    -------
    list : list, optional
        List of tuples of (tic, cadence)
    """
    TIC_ID=f"TIC {tic}"
    search_result=lk.search_lightcurve(TIC_ID, exptime=cadence, mission=mission, author=author)
    if ret_list:
        list=[]
        i=0
        for result in search_result.mission:
            sector= int(result.split(" ")[-1])
            cad = search_result[i].exptime.value[0]
            tup=(tic, sector, int(cad))
            list.append(tup)
            i+=1
        return list
    else:
        print(search_result)

def get_lightcurve(obj, cadence=None, sector=None, mission="TESS", author='SPOC'):
    """
    Searches or downloads the lightcurve.

    Searches or downloads the lightcurve data for a given TIC.

    Parameters
    ----------
    obj : TESSLC
        Lightcuvre object.
    sector : int, optional
        Observation sector, by default None.
    cadence : int, optional        
        Cadence of the observation, by default 20.
    mission : str, optional
        Observation mission, by default 'TESS'.
    author : str, optional
        Author of the data product, by default 'SPOC'.

    Attributes
    ----------
    obj.lc.full : dict
        Stores the lightcurve data.
    obj.star.ra : float
        Right ascension of the star.
    obj.star.dec : float
        Declination of the star.
    obj.star.teff : float
        Effective temperature of the star.
    obj.star.rad : float
        Radius of the star.
    obj.star.tess_mag : float
        TESS magnitude of the star.
    obj.int.sector : int
        Observation sector.
    obj.inst.cadence : float
        Cadence of the observation.
    obj.inst.cadence_err : float
        Error in the cadence time of the observation.
    obj.inst.telescope : str
        Telescope name for the observation.
    obj.inst.instrument : str
        Instrument name for the observation.
    """
    TIC_ID=f"TIC {obj.TIC}"

    if sector is None:
        search_results=lk.search_lightcurve(TIC_ID, cadence=cadence, mission=mission, author=author)
        print(search_results)
    else:
        search_lc=lk.search_lightcurve(TIC_ID, cadence=cadence, sector=sector, mission=mission, author=author)
        lc=search_lc.download(quality_bitmask=0)

        # Making the raw lightcurve file
        obj.lc.full={"time":lc['time'].value.astype(np.float64),
                       "flux":np.array(lc['flux'].value,dtype=np.float64),
                       "flux_err": np.array(lc['flux_err'].value,dtype=np.float64),
                       "quality": np.ma.getdata(lc["quality"])}

        # Assigning the stellar properties
        obj.star.ra=lc.meta['RA_OBJ']
        obj.star.dec=lc.meta['DEC_OBJ']
        obj.star.teff=lc.meta['TEFF']
        obj.star.rad=lc.meta['RADIUS']
        obj.star.tess_mag=lc.meta['TESSMAG']

        # Assigning the instrument properties
        obj.inst.sector=sector
        obj.inst.cadence=lc.meta["TIMEDEL"]
        obj.inst.telescope=lc.meta["TELESCOP"]
        obj.inst.instrument=lc.meta["INSTRUME"]
        obj.inst.cadence_err=lc.meta["TIERRELA"]