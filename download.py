import lightkurve as lk
import numpy as np

def get_lightcurve(obj, sector=None, cadence=20, mission="TESS", author='SPOC'):
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
        search_results=lk.search_lightcurve(TIC_ID, exptime=cadence, mission=mission, author=author)
        print(search_results)
    else:
        search_lc=lk.search_lightcurve(TIC_ID, sector=sector, exptime=cadence, mission=mission, author=author)
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
        obj.inst.cadence=lc.meta["TIMEDEL"]
        obj.inst.telescope=lc.meta["TELESCOP"]
        obj.inst.instrument=lc.meta["INSTRUME"]
        obj.inst.cadence_err=lc.meta["TIERRELA"]