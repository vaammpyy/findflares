from imports import *
# from defaults import *

def loadpickle(fName, sector, cadence, injrec=False):
    """
    Opens TESSLC pickled object.

    Opens TESSLC pickled object stored in <data_dir> (defined in defaults).

    Parameters
    ----------
    fName : int
        TIC-ID of the star which is also the name of the folder which has the TESSLC.
    sector : int
        Observation sector.
    cadence : int
        Cadence of the observation.
    injrec : bool, optional
        If True loads the injection recovery pickle, by default False.
    
    Returns
    -------
    Obj : TESSLC
        TESSLC object holding the lightcurve data and other parameters for flare detection process.
    """
    if injrec:
        fObj=open(f"{data_dir}/{fName}/ir_{sector}_{cadence}.pkl",'rb')
    else:
        fObj=open(f"{data_dir}/{fName}/{sector}_{cadence}.pkl",'rb')
    Obj=pickle.load(fObj)
    fObj.close()
    return Obj

def loadpickle_path(pkl_path):
    """
    Opens TESSLC pickled object.

    Opens TESSLC pickled object stored in <pkl_path> (defined in defaults).

    Parameters
    ----------
    pkl_path : str
        Path of the pickle file.
    
    Returns
    -------
    Obj : TESSLC
        TESSLC object holding the lightcurve data and other parameters for flare detection process.
    """
    fObj=open(pkl_path,'rb')
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
    data_dir : str
        Directory to store the data.

    Attributes
    ----------
    dir : str
        Path to the directory which holds the TESSLC object. Each TIC has one folder with multiple files for each sector.
    TIC : int
        TIC-ID of the star.
    flares : dict
        Stores all flare properties.
        self.flares={"t_start":, [mjd]
                     "t_peak":, [mjd]
                     "t_stop":, [mjd]
                     "i_start":, [index]
                     "i_stop":, [index]
                     "amplitude":, [e/s]
                     "duration":, [s]
                     "equi_duration":, [s]
                     "energy":}
    
    Class Attributes
    ----------------
    lc : LC
        Stores the lightcurve data and other relevant data products for flare detection
    star : STAR
        Stores the properties of the star.
    inst : INST
        Stores the instrument properties.
    """

    def __init__(self,fName, data_dir):
        """
        Initializes the TESSLC object.
        
        Parameters
        ----------
        fName : int
            TICID of the star.
        """
        self.dir=f"{data_dir}/{fName}"
        self.TIC=fName
        self.flares={"t_start":[],
                     "t_peak":[],
                     "t_stop":[],
                     "i_start":[],
                     "i_stop":[],
                     "amplitude":[],
                     "duration":[],
                     "equi_duration":[],
                     "energy":[]}

        try:
            os.mkdir(self.dir)
        except OSError:
            pass

        self.lc=self.LC()
        self.star=self.STAR()
        self.inst=self.INST()

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
        detrend_scheme : str
            Stores the algorithm used for detrending the lightcurve.
        """
        def __init__(self):
            """
            Initializes lc subclass.
            """
            self.full=None
            self.segment=None
            self.model=None
            self.detrended=None
            self.detrend_scheme=None
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
            prot : float
                Period of rotation of the star, as calculated by
                peak of the GLS, None if star is not rotating.
            dist : float
                Distance of the star cross-matched with GAIA DR3.
            """
            self.ra=None
            self.dec=None
            self.teff=None
            self.rad=None
            self.tess_mag=None
            self.prot=None
            self.dist=None
        
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
        mean=np.median(flux)
        # std=np.std(flux)
        flux_dev=abs(flux-mean)
        mad=MAD(flux)
        outlier_mask=flux_dev<1.5*mad
        gls=Gls(((time[outlier_mask], flux[outlier_mask], flux_err[outlier_mask])), fend=4, fbeg=1/14)
        fap=gls.FAP()
        pmax=gls.pmax
        pwr_lvl=gls.powerLevel(0.001)
        period=gls.best['P']

        if pmax<pwr_lvl:
            rotation= False
            print(f"PWR-DIFF::{pmax-pwr_lvl}")
        else:
            rotation= True
            print(f"PWR-DIFF::{pmax-pwr_lvl}")
        if rotation:
            print("Rotation found.")
            self.star.prot=period
            print(f"Rotation period:{self.star.prot}")
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
    
    def plot(self, mode=None, q_flags=None, segments=None, show_flares=False, show_transits=False, save_fig=False, injrec=False, injrec_run=None):
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
        plot_lightcurve(self, mode=mode, q_flags=q_flags, segments=segments, show_flares=show_flares, show_transits=show_transits, save_fig=save_fig, injrec=injrec, injrec_run=injrec_run)

    def flare_energy(self):
        """
        Evaluates flare energies and other relevant parameters.

        Attributes
        ----------
        self.flares : dict
            Stores all flare properties. First makes all the entries empty
            self.flares={"t_start":, [mjd]
                        "t_stop":, [mjd]
                        "i_start":, [index]
                        "i_stop":, [index]
                        "amplitude":, [e/s]
                        "duration":, [s]
                        "equi_duration":, [s]
                        "energy": [cgs]}
        """
        self.flares={"t_start":[],
                     "t_peak":[],
                     "t_stop":[],
                     "i_start":[],
                     "i_stop":[],
                     "amplitude":[],
                     "duration":[],
                     "equi_duration":[],
                     "energy":[]}
        get_flare_param(self)
        get_ED(self)
        get_flare_energies(self)

    def pickleObj(self):
        """
        Pickles TESSLC object.

        Pickles TESSLC object and stores it in data_dir/TICID/sector_cadence.pkl (data_dir defined in defaults).
        """
        fObj = open(f"{self.dir}/{self.inst.sector}_{int(self.inst.cadence*24*3600)}.pkl", 'wb')
        pickle.dump(self, fObj)
        fObj.close() 
        print("Pickled successfully.")
        print(f"PATH::{self.dir}/{self.inst.sector}_{int(self.inst.cadence*24*3600)}.pkl")
    
class InjRec(TESSLC):
    """
    Stores Injection recovery objects and method.

    Injection recovery class. This is supposed to be run on
    lightcurve for which pre processing has already been done.
    """

    def __init__(self, tesslc):
        """
        Initializes the InjRec object.

        Initializes the InjRec object by assigning same attributes
        from the TESSLC class.

        Parameters
        ----------
        tesslc : TESSLC
            TESS lightcurve object.
        
        Attributes
        ----------
        dir : str
            Path to the directory which holds the TESSLC object. Each TIC has one folder with multiple files for each sector.
        TIC : int
            TIC-ID of the star.
        flares : dict
            Stores all flare properties.
            self.flares={"t_start":, [mjd]
                        "t_peak":, [mjd]
                        "t_stop":, [mjd]
                        "i_start":, [index]
                        "i_stop":, [index]
                        "amplitude":, [e/s]
                        "duration":, [s]
                        "equi_duration":, [s]
                        "energy":}
        injection : dict
            Stores all injected flare parameters.
            self.injection={'t_peak':[],
                            'i_start': [],
                            'i_stop': [],
                            'ampl': [],
                            'fwhm': [],
                            'ed':[],
                            'energy':[]}
        injrec : array
            Array of injection recovery dictionaries.
            [{"injected":{"t_peak":,
                          "i_start":,
                          "i_stop":,
                          "ampl":,
                          "fwhm":,
                          "ed":,
                          "energy":},
            "recovered":{"t_start":,
                         "t_stop":,
                         "i_start":,
                         "amplitude":,
                         "duration":,
                         "equi_duration":,
                         "energy":},
            "flag":}]
        
        Class Attributes
        ----------------
        lc : LC
            Stores the lightcurve data and other relevant data products for flare detection
        orig_lc : LC
            Stores the original lightcurve data (unmodified) and other relevant data products for flare detection
        star : STAR
            Stores the properties of the star.
        inst : INST
            Stores the instrument properties.
        """
        super().__init__(tesslc.TIC)
        self.lc = copy.deepcopy(tesslc.lc)
        self.orig_lc = copy.deepcopy(tesslc.lc)
        self.star =copy.deepcopy(tesslc.star)
        self.inst =copy.deepcopy(tesslc.inst)
        self.flares =copy.deepcopy(tesslc.flares)
        self.injrec=[]
    
    def remove_flares(self):
        """
        Removes flare from lightcurve.

        Removes detected flares from the lightcurve and replaces them gaussian noise derived from the data,
        preparing the lightcurve for injection recovery test. Also, removes the previous assigned model
        and detrended information.

        Attributes
        ----------
        self.lc.full : dict
            Lightcurve dict storing lightcurve data
        self.lc.detrended : dict
            Detrended lightcurve model, assigned None
        self.lc.detrend_scheme : str
            Detrending scheme used, assigned None
        self.lc.flare : dict
            Flare information dict, assigned None
        self.lc.flare_run : bool
            Find flare run check, assigned None
        self.lc.transit : dict
            Transit information dict, assigned None
        self.lc.transit_run : bool
            Find transit run check, assigned None

        Notes
        -----
        self.lc.model not assigned to None as it's used when calculating the ED of injected flares.
        """
        print("Flare removal and LC cleaning started.")
        replace_flares_w_gaussian_noise_and_clean_attr(self)
        print("Flare removal and LC cleaning completed.")
    
    def run_injection_recovery(self, run, plot=False):
        """
        Runs injection recovery pipeline.

        Parameters
        ----------
        run : int
            Run number of the injection recovery test

        Attributes
        ----------
        self.injrec : list
            List of injtion recovery dictionaries.
        """
        print("^^^^^^^^^^^^^^")
        print(f"Inj-Rec run::{run} started.")
        self.remove_flares()
        add_flares(self, N=10)
        self.detrend_3()
        self.findflares()
        self.flare_energy()
        recover_flares(self, run)
        if plot:
            self.plot(mode="detrended", show_flares=True, show_transits=True, save_fig=True, injrec=True, injrec_run=run)
            self.plot(mode="model_overlay", save_fig=True, injrec=True, injrec_run=run)
        print(f"Inj-Rec run::{run} completed.")
        print("^^^^^^^^^^^^^^")

    def pickleObj(self):
        """
        Pickles InjRec object.

        Pickles InjRec object and stores it in data_dir/TICID/ir_sector_cadence.pkl (data_dir defined in defaults).
        """
        fObj = open(f"{self.dir}/ir_{self.inst.sector}_{int(self.inst.cadence*24*3600)}.pkl", 'wb')
        pickle.dump(self, fObj)
        fObj.close() 
        print("Pickled successfully.")
        print(f"PATH::{self.dir}/ir_{self.inst.sector}_{int(self.inst.cadence*24*3600)}.pkl")