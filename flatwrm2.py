#!/usr/bin/env /opt/tljh/user/bin/python
#!/usr/bin/env python3
# coding: utf-8


import numpy as np

from tensorflow.keras.utils import Sequence

from flare_fitter_utils import validation


class SplitGenerator(Sequence):
    '''Generator for time-series data.'''
    
    #Because... python.
    def proper_round(self, val):
        if (float(val) % 1) >= 0.5:
            x = np.ceil(val)
        else:
            x = round(val)
        return x
    
    def __init__(self, x_set, y_set, batch_size=1, length=10):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.length = length
        if self.batch_size * self.length > len(self.y):
            raise ValueError("Batch size and sample size don't match! batch_size=",self.batch_size,"length=",self.length)
    def __getitem__(self, idx):
        batch_x = np.zeros( (self.batch_size, self.length) ) * np.nan
        batch_y = np.zeros( self.batch_size ) * np.nan

        if idx > self.__len__():
            raise ValueError("Requested index too large")

        #Due to averaging the window locations with batch processing we'll introduce gaps. We add extra batches to fill the gaps. Mind the NaNs!
        if idx <= (( len(self.y) // self.batch_size ) - self.length):
            for b in np.arange(self.batch_size):
                batch_x[b]= self.x[ b * (len(self.y) // self.batch_size) + idx  : b * (len(self.y)//self.batch_size) + idx + self.length ].reshape(self.length)
                batch_y[b] = self.proper_round ( np.mean ( self.y[ b * (len(self.y) // self.batch_size) + idx  : b * (len(self.y)//self.batch_size) + idx + self.length ] ) )
        else:
            for b in np.arange(self.batch_size-1):
                try:
                    batch_x[b]= self.x[ b * (len(self.y) // self.batch_size) + idx  : b * (len(self.y)//self.batch_size) + idx + self.length ].reshape(self.length)
                    batch_y[b] = self.proper_round ( np.mean ( self.y[ b * (len(self.y) // self.batch_size) + idx  : b * (len(self.y)//self.batch_size) + idx + self.length ] ) )
                except ValueError:
                    pass
                
        return batch_x.reshape(self.batch_size, self.length,1) , batch_y.reshape( self.batch_size, 1  )
    def __len__(self):
        return ( len(self.y) // self.batch_size ) - self.length + 1 + (self.length)


def read_data(data_in, cadence=1./60/24):
    '''
    Reads light curve from file. 
    
    Parameters:
    data_in: filename with light curve, or (time, flux) tuple with the light curve
    cadence: the cadence for interpolating the light curve, in days. (default: 1 minute)
    
    Returns: 
    [t, fl, time, flux] for file input
    [t, fl] for tuple input
    where t, fl are the interpolated; time/flux are the original time and flux data
    '''
    
    from scipy.interpolate import interp1d
    from scipy.stats import median_absolute_deviation
    
    from numpy.polynomial.polynomial import Polynomial
    
    #data_in can be either a filename, or time and flux
    if type(data_in) is str:
        FILE_INPUT = True
        infile = np.genfromtxt(data_in).T
        time = infile[0]
        flux = infile[1]
    else:
        FILE_INPUT = False
        time, flux = data_in
        
    
    t = np.arange(time[0], time[-1], cadence)
    
    bad_points = np.isnan(flux)
    time=time[~bad_points]
    flux=flux[~bad_points]

    

    gaptime=time.copy()
    gapflux=flux.copy()

    allgaps = np.diff(time) > 30./60/24
    allgaps = np.where(allgaps>0)[0]


    for ii in range(len(allgaps),0,-1):
        ii-=1
        begin=time[allgaps[ii]]
        end=time[allgaps[ii]+1]
        gaplength=end-begin
        to_check_std = 0.015 #day

        um1 = (time>=time[allgaps[ii]]-to_check_std) &  (time<=time[allgaps[ii]])
        um2 = (time>=time[allgaps[ii]+1]) &  (time<=time[allgaps[ii]+1]+to_check_std)
        umplot = (time>=time[allgaps[ii]+1]-5*gaplength) &  (time<=time[allgaps[ii]+1]+5*gaplength)

        down,up = np.percentile(flux[um1],[5,95])
        beginstd = np.std( flux[um1][ (flux[um1]>=down) & (flux[um1]<=up) ] )
        down,up = np.percentile(flux[um2],[5,95])
        endstd = np.std( flux[um2][ (flux[um2]>=down) & (flux[um2]<=up) ] )
        beginmean = np.mean( flux[allgaps[ii]-5:allgaps[ii]+1] )
        endmean = np.mean( flux[allgaps[ii]+1:allgaps[ii]+5] )
        npoints = int(gaplength/np.mean(np.diff(time)))

        if beginstd<endstd:
            filling = np.random.normal(loc=0,scale=beginstd,size=npoints)
        else:
            filling = np.random.normal(loc=0,scale=endstd,size=npoints)

        timefilling  =np.linspace(begin,end,npoints+2)[1:-1]

        gaptime=np.insert(gaptime, allgaps[ii]+1, timefilling)
        gapflux=np.insert(gapflux, allgaps[ii]+1, filling+np.linspace(beginmean,endmean,npoints) )


    iflux = interp1d( gaptime, 
                     gapflux,
                     fill_value="extrapolate", 
                     bounds_error=False)
    fl = iflux(t)


    umcut = fl < np.percentile(fl,99.9)
    fl/=np.mean(fl[umcut])
    fl=(fl-1.) / np.ptp(fl[umcut])
    

    if FILE_INPUT:
        return( np.array( [t, fl, time, flux] ) )
    else:
        return( np.array( [t, fl] ) )




def prediction(data_in,  batch=1, verbose=True, DEBUG=False, GPU=0, modelfile="LSTM_weights_keplerSC.h5", save_model=False):
    '''
    Runs the prediction.
    
    Parameters:
    data_in: filename with light curve, or (time, flux) tuple with the light curve
    batch: batch number (default:1)
    verbose: enables/disables informational messages (default: True)
    DEBUG: if False, the code tries everything to suppress TensorFlow informational messages and warnings. (default: False)
    GPU: in a multi-GPU environment, tells CUDA which card to use. "-1" disables GPU usage. 
       Note: once set, this option seems to fix the GPU by CUDA, regardless what you specify later until the
       kernel is restarted (default: "0")
    modelfile: the weight file to use for the prediction
    save_model: in this mode the prediction and median-filtered prediction will be saved to a .pred file. 
       If data_in is a file, the output will be <data_in>.pred, otherwise outp.pred (default: False)

    Returns: array with predicted flare probabilities interpolated back to light curve times, 
       and an [time_sort, iflux(time_sort), pred_sort ] array containing interpolated time/flux values and the corresponding 
       predictions, as seen by the neural net.
    '''


    #Just. Shut. up.    
    import os
    import tensorflow as tf
    if not DEBUG:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["KMP_AFFINITY"] = 'noverbose'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

    import os
    GPU=str(GPU)
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
   
    if verbose:
        if (gpus == [])  and GPU!="-1":
            print('\033[91m'+'GPU '+GPU+' selected, but could not detect it. Falling back to CPU'+'\033[0m')
        elif GPU=="-1":
            print('\033[92m'+'Using CPU'+'\033[0m')
        elif len(gpus) > 0:
            print('\033[92m'+'Using GPU '+GPU+' '+'\033[0m')

    if type(data_in) is str:    
        filename = data_in
        if verbose:
            print('Processing',filename)
        lc = read_data(filename)
    else:
        t, fl = read_data(data_in)
        lc = np.array([t, fl, data_in[0], data_in[1]])

        
    X_test  = np.array([lc[1]]).T
    y_test = np.zeros_like(X_test).T
    
    window_size = 64


    generator_val = SplitGenerator(X_test, y_test.T, length=window_size, batch_size=batch)

    try:
        model = tf.keras.models.load_model(modelfile)
    except OSError:
        print('\033[91m'+'Could not find model file: '+modelfile+'  Try using the -m option, or specify an existing model file when calling the function.\nAvailable weight files:'+'\033[0m')
        import glob
        for weight in sorted(glob.glob("*.h5")):
            print(weight)
        try: 
            exit(-1)
        except NameError:
            return None
        
    pred = model.predict(generator_val, verbose=verbose)
    
    generator_time = SplitGenerator(lc[0], y_test.T, length=window_size, batch_size=batch)
    time_array = []

    for i in np.arange( generator_time.__len__() ):
        time_array.append( np.mean(generator_time[i][0], axis=1) )

    time_array=np.array(time_array).ravel()

    sort_key=np.argsort(time_array)
    pred_sort = pred[sort_key]
    pred_sort = pred_sort.reshape(pred_sort.size, )

    time_sort = np.array(time_array)[sort_key]
    pred_sort = pred_sort[np.isfinite(time_sort)]
    time_sort = time_sort[np.isfinite(time_sort)]
    

    from scipy.interpolate import interp1d
    time, flux = lc[2], lc[3]

    ipred =  interp1d(time_sort, pred_sort, bounds_error=False, fill_value=np.nan)
    iflux =  interp1d(lc[0], lc[1], bounds_error=False, fill_value=np.nan)


    if save_model:
        if type(data_in) is str:
            outfile = data_in+".pred"
        else:
            outfile = "outp.pred"
        np.savetxt(outfile, np.array( [time, flux, ipred(time)] ).T , fmt='%0.10f %0.10f %0.3f'  )

   
    return ipred(time), np.c_[time_sort, iflux(time_sort), pred_sort ]




def usage():
    print( "Usage:", sys.argv[0],"[options] <input file(s)>")
    print( "Options:")
    print( "-h, --help: print this help message")
    print( "-m, --model=<weight_file>: load the specified weight file for the prediction")
    print( "-b, --batch=<n>: number of batches to use simultaneously (default: 1)")
    print( "--gpu=<n>: set CUDA_VISIBLE_DEVICES: forces CUDA to use the given GPU. -1 disables CPU. (default: 0)")
    print( "-d, --debug: verbose mode")


#############################
###        M A I N        ###
#############################

if __name__ == "__main__":
    import getopt
    import sys
    
    import time
    import os.path

    batch=1
    debug = False
    gpu=0

    modelfile="LSTM_weights_keplerSC.h5"

    try:
        opts, args = getopt.getopt( sys.argv[1:], "hn:db:m:",\
            ["help", "debug", "batch=", "gpu=", "model="])

    except getopt.GetoptError as err:
        print( str(err) )
        usage()
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-d", "--debug"):
            debug = True
        elif opt in ("--gpu"):
            gpu = str(arg)
        elif opt in ("--model", "-m"):
            modelfile = str(arg)
        elif opt in ("-b", "--batch"):
            batch = int(arg)

    files = []

    for filename in args:
        if os.path.isfile(filename):
            files.append(filename)
        else:
            print('No such file:',filename)
            print('Exiting')
            sys.exit()
    if files == []:
        print("Hi.")
        print("(use -h for help)")
        sys.exit()

    for file in files:
        prediction(file, batch=batch, DEBUG=debug, GPU=gpu, save_model=True)
