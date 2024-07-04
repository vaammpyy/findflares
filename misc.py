import numpy as np
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

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
    # Query the TIC catalog for the star's information
    result = Catalogs.query_object(f'TIC {tic_id}', catalog='TIC')

    # Extract the distance information
    distance_pc = result[0]['d']*u.pc

    return distance_pc