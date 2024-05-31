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
            self.segmented=NONE

        #masks LC with flags array
        def mask_lc(self,flags,series_name):
            data=getattr(self,series_name)
            if len(data)>1:
                for i in range(len(data)):
                    mask=np.isin(data[i][0]["quality"],np.asarray(flags))

                    data_dict={"time":data[i][0]["time"][mask],
                               "flux":data[i][0]["flux"][mask],
                               "flux_err":data[i][0]["flux_err"][mask],
                               "quality":data[i][0]["quality"][mask],
                               "start_index":0,
                               "stop_index": len(data[i][0]["time"]),
                               "flags":flags}

                    data[i].append(data_dict)
                    setattr(self,series_name,data)
            else:
                mask=np.isin(data[0]["quality"],np.asarray(flags))
                data_dict={"time":data[0]["time"][mask],
                           "flux":data[0]["flux"][mask], 
                           "flux_err":data[0]["flux_err"][mask], 
                           "quality":data[0]["quality"][mask], 
                           "start_index":0, 
                           "stop_index": len(data[0]["time"]), 
                           "flags":flags}
                data.append(data_dict)
                setattr(self,series_name,data)

        def gls(self,series_name="full",flag_series=0):
            data=getattr(self,series_name)
            index=flag_series
            time=data[index]['time']
            flux=data[index]['flux']
            flux_err=data[index]['flux_err']
            result=xo.lomb_scargle_estimator(time, flux, yerr=flux_err, max_peaks=1, min_period=0.01, max_period=200.0, samples_per_peak=50)
            return result['peaks'][0]['period']

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
        self.lc.full=[{"time":lc['time'].value.astype(np.float64),
                       "flux":np.array(lc['flux'].value,dtype=np.float64),
                       "flux_err": np.array(lc['flux_err'].value,dtype=np.float64),
                       "quality": np.ma.getdata(lc["quality"]),
                       "start_index":0,
                       "stop_index":len(lc["time"].value),
                       "flags": np.nan}]

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
        self.lc.segmented=[]
        period=self.lc.gls(series_name="full", flag_series=1)

        t_gap= np.where(np.diff(self.lc.full[0]['time'])>self.inst.cadence+self.inst.cadence_err)[0]
        segment=[]

        for i in range(len(t_gap)):
            t_i=self.lc.full[0]['time'][t_gap[i]]
            t_i_1=self.lc.full[0]['time'][t_gap[i]+1]
            diff_t=t_i_1-t_i
            if diff_t>period:
               segment.append(t_gap[i]) 

        segment=np.insert(segment,0,-1)
        segment=np.append(segment,len(self.lc.full[0]['time'])-2)

        min_segment_length=0.0833 # defining the threshold segment lenght of two hours i.e. 1/12 [d]

        for i in range(1,len(segment)):
            segment_dict={"time":None, 
                          "flux":None, 
                          "flux_err":None, 
                          "quality":None, 
                          "start_index":None, 
                          'stop_index':None, 
                          "flags":None}
            if self.lc.full[0]['time'][segment[i]]-self.lc.full[0]['time'][segment[i-1]+1]>min_segment_length:
                segment_dict['time']=self.lc.full[0]["time"][segment[i-1]+1:segment[i]+1] 
                segment_dict['flux']=self.lc.full[0]["flux"][segment[i-1]+1:segment[i]+1] 
                segment_dict['flux_err']=self.lc.full[0]["flux_err"][segment[i-1]+1:segment[i]+1] 
                segment_dict['quality']=self.lc.full[0]["quality"][segment[i-1]+1:segment[i]+1] 
                segment_dict['start_index']=segment[i-1]+1
                segment_dict['stop_index']=segment[i]
            self.lc.segmented.append([segment_dict])