from imports import *
from defaults import *

def loadpickle(fName, sector):
    """
    Opens TESSLC pickled object.

    Opens TESSLC pickled object stored in <data_dir> (defined in defaults).

    Parameters
    ----------
    fName : int
        TIC-ID of the star which is also the name of the folder which has the TESSLC.
    sector : int
        Observation sector.
    
    Returns
    -------
    Obj : TESSLC
        TESSLC object holding the lightcurve data and other parameters for flare detection process.
    """
    fObj=open(f"{data_dir}/{fName}/{sector}.pkl",'rb')
    Obj=pickle.load(fObj)
    fObj.close()
    return Obj

class TESSLC:
    """
    Stores the TESS lightcurve.
    
    Stores TESSLC object, de-trending and flare detection routines.

    Parameter
    ---------
    fName : int
        TIC-ID of the star.

    Attributes
    ----------
    dir : str
        Path to the directory which holds the TESSLC object. Each TIC has one folder with multiple files for each sector.
    TIC : int
        TIC-ID of the star.
    
    Class Attributes
    ----------------
    lc : LC
        Stores the lightcurve data and other relevant data products for flare detection
    star : STAR
        Stores the properties of the star.
    inst : INST
        Stores the instrument properties.
    flares : FLARES
        Stores the details of the detected flares.
    """

    def __init__(self,fName):
        """
        Initializes the TESSLC object.
        
        Parameters
        ----------
        fName : int
            TICID of the star.
        """
        self.dir=f"{data_dir}/{fName}"
        self.TIC=fName

        try:
            os.mkdir(self.dir)
        except OSError:
            pass

        self.lc=self.LC()
        self.star=self.STAR()
        self.inst=self.INST()
        self.flares=self.FLARE()

    class LC:
        """
        Lightcurve data.

        Lightcurve data, segment mask, detrended data and detrending model.

        Attributes
        ----------
        full : dict
            Lightcurve data. {'time':[], 'flux': [], 'flux_err': [], 'quality': []}
        segment : ndarray
            Segment mask.
        model : ndarray
            Model flux array without the flares masked out.
        detrended : ndarray
            Detrended flux array without the flares masked out.
        flare : dict
            Dictionary storing mask (negative), start and stop times for flares.
        flare_run : bool
            Flare algorithm run check.
        transit : dict
            Dictionary storing mask (negative), start and stop times for transits.
        transit_run : bool
            Transit algorithm run check.
        """
        def __init__(self):
            """
            Initializes lc subclass.
            """
            self.full=None
            self.segment=None
            self.model=None
            self.detrended=None
            self.flare=None
            self.flare_run=False
            self.transit=None
            self.transit_run=False

    class STAR:
        def __init__(self):
            """
            Initializes the star sub-class.

            Attributes
            ----------
            ra : float
                Right ascension of the star.
            dec : float
                Declination of the star.
            teff : float
                Effective temperature of the star.
            rad : float
                Radius of the star.
            tess_mag : float
                TESS magnitude of the star.
            """
            self.ra=None
            self.dec=None
            self.teff=None
            self.rad=None
            self.tess_mag=None
        
    class INST:
        def __init__(self):
            """
            Initializes the inst sub-class.

            Attributes
            ----------
            sector : int
                Observation sector.
            cadence : float
                Cadence of the observation.
            cadence_err : float
                Error in the cadence.
            telescope : str
                Telescope for the observation.
            instrument : str
                Instrument used for the observation.
            """
            self.sector=None
            self.cadence=None
            self.telescope=None
            self.instrument=None
            self.cadence_err=None

    class FLARE:
        def __ini__(self):
            self.flares=None

    def download_lc(self,sector, cadence=None, mission='TESS', author="SPOC", segment=False, clean=False):
        """
        Downloads the TESS lightcurve data for the star.

        Parameters
        ----------
        sector : int
            Observation sector.
        cadence : int, optional        
            Cadence of the observation, by default None.
        mission : str, optional
            Observation mission, by default 'TESS'.
        author : str, optional
            Author of the data product, by default 'SPOC'.
        segment : bool, optional
            Set to True to generate the segment mask, by default False.
        clean : bool, optional
            Set to True to clean the lightcurve, by default False.
        """
        print("Downloading started.")
        if cadence == None:
            get_lightcurve(self, sector=sector, mission=mission, author=author)
        else: 
            get_lightcurve(self, cadence=cadence, sector=sector, mission=mission, author=author)
        print("Downloading completed.")
        if clean:
            print("Cleaning started.")
            self.clean_lc()
            print("Cleaning completed.")
        if segment:
            print("Segmentation started.")
            self.segment_lc()
            print("Segmentation completed.")
    
    def search_lc(self,cadence=20, mission="TESS", author='SPOC'):
        """
        Searches for lightcurve.

        Searches for lightcuvre for a given TIC-ID and given cadence.

        Parameters
        ----------
        cadence : int, optional
            Cadence of the observation, by default 20.
        mission : str, optional
            Mission of the observation, by default "TESS".
        author : str, optional
            Author of the data product, by default 'SPOC'.
        """
        get_lightcurve(self, cadence=cadence, mission=mission, author="SPOC")

    def segment_lc(self):
        """
        Segments the lightcurve.

        Segments the lightcurve over the data gaps that are larger than <factor>*time period of the stellar rotation.
        No segment shorter than <min_segment_len> are taken as the data is not sufficient for a decent GP regression.
        """
        mask=get_mask(self, q_flags=[0])
        period=get_period(self, mask=mask)
        segment_lightcurve(self, period=period, factor=1, min_segment_len=2)
    
    def clean_lc(self):
        """
        Cleans the lightcurve.
        
        Removes nans and data points with q_flag other than [0,512] from all the columns in the lightcurve data.
        """
        clean_lightcurve(self)
    
    def detrend(self, segments=None, iter_detrend=True, mask_transit=True):
        """
        Detrends lightcurve.

        Detrends lightcurve using gaussian process regression if rotation is found else uses a median filter of 12hr window.

        Parameters
        ----------
        segments : list, optional
            Segments to detrend, by default None and it'll detrend all the segments.
        iter_detrend : bool, optional
            If True iterative detrending will be performed using the flare masked method, by default True.
        mask_transit : bool, optional
            If True then in the second iteration of detrending process transits will be masked, by default True.
        """
        rotation=check_rotation(self)
        if rotation:
            GaussianProcess_detrend(self, segments=segments)
            if iter_detrend:
                find_flare(self, find_transit=mask_transit)
                GaussianProcess_detrend(self, segments=segments, mask_flare=True, mask_transit=mask_transit)
        else:
            Median_detrend(self)

    def detrend_2(self, segments=None, iter_detrend=True, mask_transit=True):
        """
        Detrends lightcurve, 2 iteration of GP.

        Detrends lightcurve using median filter then checks for rotation in the residual,
        if FAP<0.01 then rotation is found and GP is run for detrending else just median filter with
        12hr window. 

        Parameters
        ----------
        segments : list, optional
            Segments to detrend, by default None and it'll detrend all the segments.
        iter_detrend : bool, optional
            If True iterative detrending will be performed using the flare masked method, by default True.
        mask_transit : bool, optional
            If True then in the second iteration of detrending process transits will be masked, by default True.
        """
        print("Detrending started.")
        Median_detrend(self)
        rotation=check_rotation_2(self)
        if rotation:
            print("Rotation found.")
            print("Segmentation started.")
            self.segment_lc()
            print("Segmentation completed.")
            print("iter 1")
            GaussianProcess_detrend(self, segments=segments, mask_outlier=True)
            if iter_detrend:
                find_flare(self, find_transit=mask_transit)
                print("iter 2")
                GaussianProcess_detrend(self, segments=segments, mask_flare=True, mask_transit=mask_transit, mask_outlier=True)
        else:
            print("Rotation not found.")
            # if iter_detrend:
            #     find_flare(self, find_transit=mask_transit)
            #     print("iter 2")
            #     Median_detrend(self, mask_flare=False, mask_transit=False)
        print("Detrending completed.")

    def detrend_3(self, segments=None, iter_detrend=True, mask_transit=True):
        """
        Detrends lightcurve, adaptive iteration of GP.

        Detrends lightcurve using median filter then checks for rotation in the residual,
        if FAP<0.01 then rotation is found and GP is run for detrending else just median filter with
        12hr window. 

        Parameters
        ----------
        segments : list, optional
            Segments to detrend, by default None and it'll detrend all the segments.
        iter_detrend : bool, optional
            If True iterative detrending will be performed using the flare masked method, by default True.
        mask_transit : bool, optional
            If True then in the second iteration of detrending process transits will be masked, by default True.
        """
        print("Detrending started.")
        Median_detrend(self)

        #checking for rotation
        time=self.lc.detrended['time']
        flux=self.lc.detrended['flux']
        flux_err=self.lc.detrended['flux_err']
        mean=np.mean(flux)
        std=np.std(flux)
        flux_dev=abs(flux-mean)
        outlier_mask=flux_dev<3*std
        gls=Gls(((time[outlier_mask], flux[outlier_mask], flux_err[outlier_mask])), fend=4, fbeg=1/14)
        fap=gls.FAP()

        if fap>0.001:
            rotation= False
        else:
            rotation= True

        rotation=check_rotation_2(self)
        if rotation:
            print("Rotation found.")
            print("Segmentation started.")
            self.segment_lc()
            print("Segmentation completed.")
            GaussianProcess_detrend(self, mask_flare=True, mask_transit=mask_transit, mask_outlier=True, iter=True)
        else:
            print("Rotation not found.")
            # if iter_detrend:
            #     find_flare(self, find_transit=mask_transit)
            #     print("iter 2")
            #     Median_detrend(self, mask_flare=False, mask_transit=False)
        print("Detrending completed.")
    
    def findflares(self):
        """
        Finds flares in lightcurve.
        """
        print("Find flares started.")
        find_flare(self)
        print("Find flares completed.")
    
    def plot(self, mode=None, q_flags=None, segments=None, show_flares=False, show_transits=False, save_fig=False):
        """
        Plots lightcurve.

        Parameter
        ---------
        mode : str, optional
            Plotting mode, by default None.
            None : Full lightcurve
            'model_overlay' : Full lightcurve with model overlaid.
            'detrended' : Detrended lightcurve.
            'flare_zoom' : Zoomed in flare.
        q_flags : list, optional
            Quality flags of the data to be plotted, by default None.
        segments : list, optional
            Segment of the data to be plotted, by default None.
        show_flares : bool, optional
            If True flares are plotted in red, by default False.
        show_transits : bool, optional
            If True transits are plotted in blue, by default False.
        save_fig : bool, optional
            If True figures are saved, by default False.
        """
        plot_lightcurve(self, mode=mode, q_flags=q_flags, segments=segments, show_flares=show_flares, show_transits=show_transits, save_fig=save_fig)

    def pickleObj(self):
        """
        Pickles TESSLC object.

        Pickles TESSLC object and stores it in data_dir/TICID/sector.pkl (data_dir defined in defaults).
        """
        fObj = open(f"{self.dir}/{self.inst.sector}_{int(self.inst.cadence*24*3600)}.pkl", 'wb')
        pickle.dump(self, fObj)
        fObj.close() 
        print("Pickled successfully.")
        print(f"PATH::{self.dir}/{self.inst.sector}_{int(self.inst.cadence*24*3600)}.pkl")