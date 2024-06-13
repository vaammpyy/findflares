from imports import *
from defaults import *


def loadpickle(fName):
    fObj=open(f"{data_dir}/{fName}/TESSLC",'rb')
    Obj=pickle.load(fObj)
    fObj.close()
    return Obj

class TESSLC:

    def __init__(self,fName):
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

    def pickleObj(self):
        fObj = open(f"{self.dir}/TESSLC", 'wb')
        pickle.dump(self, fObj)
        fObj.close() 

    class LC:
        def __ini__(self):
            self.full=NONE
            self.full_flare_masked=NONE
            self.segment=NONE
            self.model=NONE
            self.model_2=NONE
            self.detrended=NONE
            self.detrended_2=NONE
            self.flare=None
            self.flare_2=NONE

        # Cleans and removes all the data points with quality flags other than [0,512], and the nan values from the data.  
        def clean_lc(self):

            data=self.full
            time=data['time']
            flux=data['flux']
            flux_err=data['flux_err']
            quality=data['quality']
            finite_time_mask=np.isfinite(time)
            finite_flux_mask=np.isfinite(flux)
            finite_flux_err_mask=np.isfinite(flux_err)
            finite_quality_mask=np.isfinite(quality)

            finite_mask=finite_flux_mask & finite_time_mask & finite_flux_err_mask & finite_quality_mask

            quality_mask=self.get_mask(q_flags=[0,512])

            combined_mask= finite_mask & quality_mask

            self.full=    {'time': time[combined_mask],
                                'flux': flux[combined_mask],
                                'flux_err': flux_err[combined_mask],
                                'quality': quality[combined_mask]} 

            self.segment=  self.segment[combined_mask]

        #Calculates and return the period of the LC with the given masks.
        def gls(self,mask):
            data=self.full
            time=data['time'][mask]
            flux=data['flux'][mask]
            flux_err=data['flux_err'][mask]
            result=xo.lomb_scargle_estimator(time, flux, yerr=flux_err, max_peaks=1, min_period=0.01, max_period=200.0, samples_per_peak=50)
            return result['peaks'][0]['period']
        
        #masks LC with flags array
        def get_mask(self, q_flags=None, segments=None):
            
            if q_flags is None:
                q_mask=np.ones_like(self.full['time'], dtype=bool)
            else:
                q_mask=np.isin(self.full['quality'],np.asarray(q_flags))

            if segments is None:
                seg_mask=np.ones_like(self.full['time'], dtype=bool)
            else:
                seg_mask=np.isin(self.segment,np.asarray(segments))

            mask= q_mask & seg_mask

            return(mask)

        # Gaussian Process de-trending methods.
        # This internal method that takes in the model_mask (mask to be considered when modelling the LC) and xeval_mask (mask to evaluate the model at) and returns the map_solution.
        def _QLC_model(self, model_mask, eval_mask):

            x=self.full['time'][model_mask]
            y=self.full['flux'][model_mask]
            yerr=self.full['flux_err'][model_mask]

            x_eval=self.full['time'][eval_mask]

            period_peak=self.gls(mask=model_mask)

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
                
        # Performs GP detrending on all the segments or an array of chosen segments and stores the result in self.lc.detrended.
        # Fitting is performed on the points with q_flag=0
        def GP_detrend(self, segments=None):

            q_flags=[0]

            lc_detrended={'time':np.array([]),
                          "flux":np.array([]),
                          "flux_err":np.array([]),
                          "quality":np.array([])}

            model_lc={'time':np.array([]),
                      "flux":np.array([])}

            if segments is None:
                segments=np.unique(self.segment)
            
            print(f"GP modelling segments: {segments}")
            for seg in segments:
                print()
                print("------------------------------------")
                print(f"Segment: {seg}, started.")
                model_mask=self.get_mask(q_flags=q_flags, segments=[seg])
                eval_mask=self.get_mask(segments=[seg])
                map_soln=self._QLC_model(model_mask=model_mask, eval_mask=eval_mask)

                lc_detrended['time']= np.append(lc_detrended['time'],self.full['time'][eval_mask])
                lc_detrended['flux']= np.append(lc_detrended['flux'],self.full['flux'][eval_mask]-map_soln['pred'])
                lc_detrended['flux_err']= np.append(lc_detrended['flux_err'],self.full['flux_err'][eval_mask])
                lc_detrended['quality']= np.append(lc_detrended['quality'],self.full['quality'][eval_mask])

                model_lc['time']= np.append(model_lc['time'],self.full['time'][eval_mask])
                model_lc['flux']= np.append(model_lc['flux'],map_soln['pred'])
                print(f"Segment: {seg}, completed.")
            self.detrended=lc_detrended
            self.model=model_lc


        def iterative_detrend(self, segments=None):

            q_flags=[0]

            lc_detrended={'time':np.array([]),
                          "flux":np.array([]),
                          "flux_err":np.array([]),
                          "quality":np.array([])}

            model_lc={'time':np.array([]),
                      "flux":np.array([])}

            if segments is None:
                segments=np.unique(self.segment)

            print(f"Iteration 1: GP modelling segments: {segments}")
            for seg in segments:
                print()
                print("------------------------------------")
                print(f"Segment: {seg}, started.")
                model_mask=self.get_mask(q_flags=q_flags, segments=[seg])
                eval_mask=self.get_mask(segments=[seg])
                map_soln=self._QLC_model(model_mask=model_mask, eval_mask=eval_mask)

                lc_detrended['time']= np.append(lc_detrended['time'],self.full['time'][eval_mask])
                lc_detrended['flux']= np.append(lc_detrended['flux'],self.full['flux'][eval_mask]-map_soln['pred'])
                lc_detrended['flux_err']= np.append(lc_detrended['flux_err'],self.full['flux_err'][eval_mask])
                lc_detrended['quality']= np.append(lc_detrended['quality'],self.full['quality'][eval_mask])

                model_lc['time']= np.append(model_lc['time'],self.full['time'][eval_mask])
                model_lc['flux']= np.append(model_lc['flux'],map_soln['pred'])
                print(f"Segment: {seg}, completed.")
            self.detrended=lc_detrended
            self.model=model_lc

            lc_detrended={'time':np.array([]),
                          "flux":np.array([]),
                          "flux_err":np.array([]),
                          "quality":np.array([])}

            model_lc={'time':np.array([]),
                      "flux":np.array([])}

            if segments is None:
                segments=np.unique(self.segment)
            
            self.find_flare()

            print(f"Iteration 2: GP modelling segments [flare masked]: {segments}")
            for seg in segments:
                print()
                print("------------------------------------")
                print(f"Segment: {seg}, started.")
                model_mask=self.get_mask(q_flags=q_flags, segments=[seg])
                flare_mask=self.flare

                comb_model_mask= flare_mask & model_mask

                eval_mask=self.get_mask(segments=[seg])
                map_soln=self._QLC_model(model_mask=comb_model_mask, eval_mask=eval_mask)

                lc_detrended['time']= np.append(lc_detrended['time'],self.full['time'][eval_mask])
                lc_detrended['flux']= np.append(lc_detrended['flux'],self.full['flux'][eval_mask]-map_soln['pred'])
                lc_detrended['flux_err']= np.append(lc_detrended['flux_err'],self.full['flux_err'][eval_mask])
                lc_detrended['quality']= np.append(lc_detrended['quality'],self.full['quality'][eval_mask])

                model_lc['time']= np.append(model_lc['time'],self.full['time'][eval_mask])
                model_lc['flux']= np.append(model_lc['flux'],map_soln['pred'])
                print(f"Segment: {seg}, completed.")
            self.detrended_2=lc_detrended
            self.model_2=model_lc

        # Merges the flares which are closer than the close_th.
        # Returns the merged start and stop indices
        def _merge_flares(self, start_indices, stop_indices, close_th):
            # Check if the input arrays are valid
            if len(start_indices) != len(stop_indices):
                raise ValueError("Start and stop indices arrays must have the same length")
            
            # Initialize the lists for new start and stop indices
            new_start_indices = []
            new_stop_indices = []
            
            # Initialize the first event
            current_start = start_indices[0]
            current_stop = stop_indices[0]
            
            for i in range(1, len(start_indices)):
                next_start = start_indices[i]
                next_stop = stop_indices[i]
                
                # Check if the difference between the current stop and next start is less than or equal to close_threshold 
                if next_start - current_stop <= close_th:
                    # Merge events by updating the current stop to the next stop
                    current_stop = next_stop
                else:
                    # Append the current event to the new lists
                    new_start_indices.append(current_start)
                    new_stop_indices.append(current_stop)
                    # Update the current event to the next event
                    current_start = next_start
                    current_stop = next_stop
            
            # Append the last event to the new lists
            new_start_indices.append(current_start)
            new_stop_indices.append(current_stop)
            
            return new_start_indices, new_stop_indices

        # Detects the tail of the flares which are above the sig_lvl
        # Returns the new start and stop indices including the flare tail
        def _include_tail(self, data, start_indices, stop_indices, sig_lvl):
            # Calculate mean and standard deviation of the data
            mean = np.mean(data)
            std_dev = np.std(data)
            
            # Determine the threshold
            threshold = mean + sig_lvl * std_dev
            
            # Initialize new start and stop indices
            new_start_indices = []
            new_stop_indices = []

            for start, stop in zip(start_indices, stop_indices):
                new_start = start
                new_stop = stop
                
                # Adjust start index backwards
                for i in range(start - 1, -1, -1):
                    if data[i] > threshold:
                        new_start = i
                    else:
                        break
                
                # Adjust stop index forwards
                for i in range(stop + 1, len(data)):
                    if data[i] > threshold:
                        new_stop = i
                    else:
                        break
                
                new_start_indices.append(new_start)
                new_stop_indices.append(new_stop)
            
            return new_start_indices, new_stop_indices

        # Find flare according to the modified chang, davenport and illin method.
        # returns the start and stop of the flares.
        def find_flare(self):
            # data=getattr(self.lc, 'detrended', None)
            data=self.detrended
            flux=data['flux']
            flux_err=data['flux_err']
            start, stop=FINDflare(flux, flux_err, N1=3, N2=1, N3=3, avg_std=True, std_window=5)
            start, stop= self._include_tail(flux, start, stop, sig_lvl=1)
            start, stop=self._merge_flares(start, stop, close_th=9)

            #making a positive flare mask
            flare_mask=np.ones(len(self.segment), dtype=bool)
            for i in range(len(start)):
                flare_mask[start[i]:stop[i]+1]=0
            self.flare=flare_mask

            return start, stop

        # Find flare according to the modified chang, davenport and illin method.
        # returns the start and stop of the flares.
        def find_flare_idetrend(self):
            # data=getattr(self.lc, 'detrended', None)
            data=self.detrended_2
            flux=data['flux']
            flux_err=data['flux_err']
            start, stop=FINDflare(flux, flux_err, N1=3, N2=1, N3=3, avg_std=True, std_window=3)
            start, stop= self._include_tail(flux, start, stop, sig_lvl=1)
            start, stop=self._merge_flares(start, stop, close_th=9)

            #making a positive flare mask
            flare_mask=np.ones(len(self.segment), dtype=bool)
            for i in range(len(start)):
                flare_mask[start[i]:stop[i]+1]=0
            self.flare=flare_mask

            return start, stop

    class STAR:
        pass
    class INST:
        pass
    class FLARE:
        def __ini__(self):
            self.flares=None


    # Downloads the LC and stores relevant meta information.
    def download_lc(self,sector,cadence=20, mission='TESS', author="SPOC"):
        TIC_ID=f"TIC {self.TIC}"
        search_lc=lk.search_lightcurve(TIC_ID, sector=sector, exptime=cadence, mission=mission, author="SPOC")
        lc=search_lc.download(quality_bitmask=0)

        # Making the raw lightcurve file
        self.lc.full={"time":lc['time'].value.astype(np.float64),
                       "flux":np.array(lc['flux'].value,dtype=np.float64),
                       "flux_err": np.array(lc['flux_err'].value,dtype=np.float64),
                       "quality": np.ma.getdata(lc["quality"])}

        # Assigning the stellar properties
        self.star.ra_obj=lc.meta['RA_OBJ']
        self.star.dec_obj=lc.meta['DEC_OBJ']
        self.star.teff=lc.meta['TEFF']
        self.star.rad=lc.meta['RADIUS']
        self.star.tess_mag=lc.meta['TESSMAG']

        # Assigning the instrument properties
        self.inst.cadence=lc.meta["TIMEDEL"]
        self.inst.telescope=lc.meta["TELESCOP"]
        self.inst.instrument=lc.meta["INSTRUME"]
        self.inst.cadence_err=lc.meta["TIERRELA"]
    
    # Searching for lightcurve data
    def search_lc(self,cadence=20, author='SPOC'):
        TIC_ID=f"TIC {self.TIC}"
        search_results=lk.search_lightcurve(TIC_ID, exptime=cadence, mission="TESS", author=author)
        print(search_results)

    #Segmenting LC with conditions, t_gap>period, t_segment>2hr
    def segment_lc(self):
        # self.lc.segmented=[]
        mask=self.lc.get_mask(q_flags=[0])
        period=self.lc.gls(mask)

        t_gap= np.where(np.diff(self.lc.full['time'])>self.inst.cadence+self.inst.cadence_err)[0]

        segment=[]

        for i in range(len(t_gap)):
            t_i=self.lc.full['time'][t_gap[i]]
            t_i_1=self.lc.full['time'][t_gap[i]+1]
            diff_t=t_i_1-t_i
            if diff_t>period:
                segment.append(t_gap[i]) 

        segment=np.insert(segment,0,-1)
        segment=np.append(segment,len(self.lc.full['time'])-1)

        min_segment_length=0.0833 # defining the threshold segment length of two hours i.e. 1/12 [d]

        segment_mask=np.zeros(len(self.lc.full['time']))

        segment_number=1
        for i in range(1,len(segment)):
            if self.lc.full['time'][segment[i]]-self.lc.full['time'][segment[i-1]+1]>min_segment_length:
                segment_mask[segment[i-1]+1:segment[i]+1]=segment_number
                segment_number+=1
        
        self.lc.segment=segment_mask

    # Adds a single flare in the self.flares.lc
    def add_flare1(self,tpeak,fwhm,ampl,q_flags=None,segments=None):
        mask=self.lc.get_mask(q_flags=q_flags, segments=segments)
        t=self.lc.full['time'][mask]
        flux=self.lc.full['flux'][mask]
        flux_err=self.lc.full['flux_err'][mask]
        quality=self.lc.full['quality'][mask]
        flare=aflare1(t,tpeak=tpeak, fwhm=fwhm, ampl=ampl)
        self.flares.lc={"time":t, "flux":flux+flare,
                        "flux_err":flux_err, "quality":quality}