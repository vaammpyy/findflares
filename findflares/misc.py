import numpy as np
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from time import time

Gaia.TIMEOUT = 60

def MAD(x):
    """
    Median absolute deviation.

    Calculates median absolute deviation of any array.
    median(x_i-mean_x)

    Parameters
    ----------
    x : np.array
        Array to calculate MAD of.
    
    Returns
    -------
    MAD : float
        Mean absolute deviation of the array.
    """
    mean=np.mean(x)
    abs_dev=abs(x-mean)
    mad=np.median(abs_dev)
    return mad

def get_dist_gaia(ra, dec):
    """
    DEPRECATED

    Gets distance of the star from GAIA.

    Gets distance of the star from GAIA catalogue if the star is found at the given ra and dec.

    Parameters
    ----------
    ra : float
        Right ascension of the star.
    dec : float
        Declination of the star.
    
    Returns
    -------
    dist : float, None
        Distance in parsecs if found in the catalogue.
    """
    print("Fetching distance from GAIA.")
    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')

    # Perform the crossmatch with Gaia DR3
    width = u.Quantity(5, u.arcsec)
    height = u.Quantity(5, u.arcsec)

    result = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    # Fetch Parallax Data
    if len(result) > 0:
        parallax_arcsec = result['parallax'][0]/1000
        print(f"Parallax: {parallax_arcsec} arcsec")
        print(f"Distance: {1/parallax_arcsec} pc")
        dist=1/parallax_arcsec*u.pc
        return dist
    else:
        print("No match found in Gaia DR3.")
        return None
    
def get_dist_tess(tic_id):
    """
    Gets distance of the star from TESS.

    Gets distance of the star from TESS catalogue if the star is found at the given TIC-ID.

    Parameters
    ----------
    tic_id : int
        TIC-ID of the star.
    
    Returns
    -------
    dist : float, None
        Distance in parsecs.
    """
    print("PIPELINE::STEP::Fetching distance from TESS.")
    # Query the TIC catalog for the star's information
    start_time=time()
    result = Catalogs.query_object(f'TIC {tic_id}', catalog='TIC')
    stop_time=time()

    # Extract the distance information
    if len(result)>0:
        distance_pc = result[0]['d']*u.pc
        print("PIPELINE::STEP::Distance found.", flush=True)
        print(f"META::DISTANCE::{distance_pc}", flush=True)
        print(f"PIPELINE::TIME::{stop_time-start_time:.2f} s", flush=True)
        return distance_pc
    else:
        print("PIPELINE::STEP::Distance not found.")
        print(f"PIPELINE::TIME::{stop_time-start_time:.2f} s", flush=True)
        return None
