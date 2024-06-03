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
            self.segment=NONE


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

    class STAR:
        pass
    class INST:
        pass
    class FLARE:
        pass

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

    # Masks the LC with quality flags (numpy array).
    # def mask_lc(self,flags):
    #     mask=np.isin(self.lc.full['quality'],np.asarray(flags))
    #     self.lc.masked={"time":self.lc.full[0]['time'][mask], "flux":self.lc.full[0]['flux'][mask], "flux_err": self.lc.full[0]['flux_err'][mask], "quality": self.lc.full[0]["quality"][mask]}
    #     self.lc.mask_quality=flags

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

        min_segment_length=0.0833 # defining the threshold segment lenght of two hours i.e. 1/12 [d]

        segment_mask=np.zeros(len(self.lc.full['time']))

        segment_number=1
        for i in range(1,len(segment)):
            if self.lc.full['time'][segment[i]]-self.lc.full['time'][segment[i-1]+1]>min_segment_length:
                segment_mask[segment[i-1]+1:segment[i]+1]=segment_number
                segment_number+=1
        
        self.lc.segment=segment_mask


    def add_flare1(self,tpeak,fwhm,ampl,q_flags=None,segments=None):
        mask=self.lc.get_mask(q_flags=q_flags, segments=segments)
        t=self.lc.full['time'][mask]
        flux=self.lc.full['flux'][mask]
        flux_err=self.lc.full['flux_err'][mask]
        quality=self.lc.full['quality'][mask]
        flare=aflare1(t,tpeak=tpeak, fwhm=fwhm, ampl=ampl)
        self.flares.lc={"time":t, "flux":flux+flare,
                        "flux_err":flux_err, "quality":quality}