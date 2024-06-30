import numpy as np

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