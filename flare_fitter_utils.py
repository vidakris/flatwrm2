import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from numba import jit
from scipy.ndimage import median_filter
from scipy.integrate import simps
from astropy import units as u
import warnings
from tqdm import tqdm

@jit(fastmath=True,nopython=True,nogil=True,cache=True)
def f1(x,tpeak,fwhm):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]

    return _fr[0]+                                   _fr[1]*((x-tpeak)/fwhm)+                  _fr[2]*((x-tpeak)/fwhm)**2.+              _fr[3]*((x-tpeak)/fwhm)**3.+              _fr[4]*((x-tpeak)/fwhm)**4.

@jit(fastmath=True,nopython=True,nogil=True,cache=True)
def f2(x,tpeak,fwhm):
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    return _fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +            _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] )

@jit(fastmath=True,nopython=True,nogil=True,cache=True)
def aflare1_fast(t, tpeak, fwhm, ampl):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723

    Use this function for fitting classical flares with most curve_fit
    tools.

    Note: this model assumes the flux before the flare is zero centered

    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare

    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''

    flare = np.zeros_like(t)
    flare[(t<= tpeak) * ((t-tpeak)/fwhm > -1.)] += f1(t[(t<= tpeak) * ((t-tpeak)/fwhm > -1.)] ,tpeak,fwhm)
    flare[t > tpeak] += f2(t[t > tpeak],tpeak,fwhm)
    flare *= np.abs(ampl)

    return flare

@jit(fastmath=True,nopython=True,nogil=True,cache=True)
def diff(params, time,y):
    tpeak, fwhm, ampl = params[0],params[1],params[2]
    return aflare1_fast(time,tpeak, fwhm, ampl) - y

@jit(fastmath=True,nopython=True,nogil=True,cache=True)
def diff_linear_nopeak2(params, time,y):
    coefficients = params[:-3][::-1]
    model1 = np.zeros_like(time)
    i = 0
    for coefficient in range(len(coefficients)):
        model1 += coefficients[i] * time**i
        i = i+1

    tpeak, fwhm, ampl = params[-3],params[-2],params[-1]
    model2 = aflare1_fast(time,tpeak, fwhm, ampl)
    return  model1+model2 - y

@jit(fastmath=True,nopython=True,nogil=True,cache=True)
def diff_linear_nopeak_multiflare(params, time,y, nflares=1):
    nflares = 3*nflares
    coefficients = params[:-nflares][::-1]
    model = np.zeros_like(time)
    i = 0
    for coefficient in range(len(coefficients)):
        model += coefficients[i] * time**i
        i = i+1

    for fl in np.arange(0,nflares,3)[::-1]:
        tpeak, fwhm, ampl = params[-(fl+3)],params[-(fl+2)],params[-(fl+1)]
        model += aflare1_fast(time, tpeak, fwhm, ampl)

    return  model - y

def get_bkg_poly_coeffs(t,y,order,debug=False):
    if debug: print('Getting bkg iteratively...')
    z = np.polyfit( t, y, order )

    residualtest = y- np.poly1d(z)(t)
    npoints = 0
    while True:
        umtest = residualtest<0.5*np.std(residualtest)
        try:
            z = np.polyfit( t[umtest],  y[umtest], order )
        except TypeError:
            # Probably too low number of points
            z = np.polyfit( t,   residualtest, order )
            break
        residualtest = y-np.poly1d(z)(t)
        if npoints==len(umtest):
            # Break if no more pts above 0.5 std
            break
        npoints=len(umtest)
    del residualtest
    del npoints

    return z

def get_BIC(chi2,n,k):
    return n*np.log(chi2/n) + k*np.log(n)

def fit_bkg_and_flare_model(lc,umshort,umwide,order=2,debug=False):
    # ---- Fit polynomial iteratively to get bkg ----------------
    with warnings.catch_warnings(record=True):
        z = get_bkg_poly_coeffs( lc[umwide,0], lc[umwide,1], order, debug=debug)
    pol = np.poly1d(z)

    # ---- Remove bkg and estimate initial flare parameters ----------------
    linfit = pol(lc[umwide,0])
    linfitshort = pol(lc[umwide,0]) ######"""umshort"""
    # Elliminate 1 pt outliers before finding maximum
    convy = median_filter(lc[umwide,1]-linfitshort,3,mode='mirror') ###"""umshort"""

    amp0 = lc[umwide,1]-linfitshort  ###"""umshort"""
    down = np.percentile(amp0,5)
    amp0 = np.ptp(amp0[amp0>=down])
    if amp0==0:
        amp0 = np.ptp(lc[umwide,1]-linfit)
        if amp0==0:
            amp0 = np.ptp(lc[:,1])
    fwhm0 = 5./60./24.
    tpeak0max = lc[umwide,0][ np.argmax(convy) ] ###"""umshort"""

    # ---- Get lower bounds of poly+flare model fitting ----------------
    lowbound = []
    for coefficient in z[:-1]:
        lowbound.append( coefficient-np.abs(coefficient)*0.002  )
    lowbound = tuple(lowbound)
    lowbound += (z[-1]-np.abs(z[-1])*0.0001,  lc[umwide,0].min(),2/(24*60),0 )

    # ---- Get upper bounds of poly+flare model fitting ----------------
    upbound = []
    for coefficient in z:
        upbound.append( coefficient+np.abs(coefficient)*0.002  )
    upbound = tuple(upbound)
    upbound += ( lc[umwide,0].max(),np.inf,10*amp0  )

    # ---- Fit polynomial and flare at once ----------------
    res = least_squares(diff_linear_nopeak2, [*z,tpeak0max,fwhm0,amp0],
                        bounds=( lowbound , upbound ),
                        args=(lc[umwide,0], lc[umwide,1]))

    return res,convy,linfit,linfitshort,tpeak0max,fwhm0,amp0,z

def fit_bkg_and_multi_flare_model(lc,um,testparams,nflares=1,debug=0):
    nflareparams = 3*nflares

    # ---- Get lower bounds of poly+flare model fitting ----------------
    lowbound = []
    for coefficient in testparams[:-nflareparams-1]:
        lowbound.append( coefficient-np.abs(coefficient)*0.001  )
    lowbound.append( testparams[-nflareparams-1]-np.abs(testparams[-nflareparams-1])*0.001 )
    for fl in reversed(range(0,nflareparams,3)):
        tpeak, fwhm, ampl = testparams[-(fl+3)],testparams[-(fl+2)],testparams[-(fl+1)]
        lowbound += [ tpeak-0.1,1/(24*60), 0 ]
    lowbound = tuple(lowbound)

    # ---- Get upper bounds of poly+flare model fitting ----------------
    upbound = []
    for coefficient in testparams[:-nflareparams]:
        upbound.append( coefficient+np.abs(coefficient)*0.001  )
    for fl in reversed(range(0,nflareparams,3)):
        tpeak, fwhm, ampl = testparams[-(fl+3)],testparams[-(fl+2)],testparams[-(fl+1)]
        upbound += [ tpeak+0.1, np.inf, 10*ampl ]
    upbound = tuple(upbound)

    # ---- Fit polynomial and flare at once ----------------
    if debug: print('Fitting complex flare components simultaneously')

    res = least_squares(diff_linear_nopeak_multiflare, testparams,
                        bounds=( lowbound , upbound ),
                        args=(lc[um,0], lc[um,1],nflares))

    return res

def get_flares(lc, pred_probability_cut=0.5,
               secondaryflare_sigma=1.5,
               progressbar=True,
               inputdata=False,
               whichflare=None,
               flaretime=None,
               stella=False,
               plotting=False,
               debug=False):

    # Get flare positions
    um = lc[:,2]>pred_probability_cut
    if um[0]:
        # if first point is flare add 0 to jumps
        um = np.diff(um)
        um = np.where(um == 1)[0]
        um = np.concatenate(( np.atleast_1d(0) , um ))
    else:
        um = np.diff(um)
        um = np.where(um == 1)[0]

    flare_epsilon_all = []
    flare_duration_all = []
    modeltpeak_all = []
    modelamp_all = []
    tpeak_all = []
    amp_all = []
    SperN_all = []

    for ii in (tqdm(range(0,len(um),2)) if progressbar else range(0,len(um),2)):
        if whichflare is not None and flaretime is None and ii!=whichflare*2: continue
        if flaretime is not None and ~(lc[um[ii],0]<=flaretime<=lc[um[ii+1],0]): continue
        if debug: print('#'*30,'flare',ii//2,'#'*30)

        try:
            dt = lc[um[ii+1],0]-lc[um[ii],0]
            cut1 = um[ii]
            cut2 = um[ii+1]
        except IndexError:
            dt = lc[-1,0]-lc[um[ii],0]
            cut1 = um[ii]
            cut2 = -1

        # ---- Exclude flares at edges and flares w/ large internal gaps ----------------
        if not inputdata and lc[cut2+1,0]-lc[cut2,0] > np.mean(np.diff(lc[:,0]))*30 and dt<=6./60/24:
            if debug: print('Skipping short flare at gap')
            # if short flare is at gap -> remove because it is a FP
            continue

        if not inputdata and np.any( np.diff(lc[cut1:cut2,0]) > dt*0.5 ):
            # if flare has a large gap -> remove because it is a FP
            # check lc length before and after gap
            gap_at = np.argmax(  np.diff(lc[cut1:cut2,0]) > dt*0.5  )
            lcbeforegap = lc[cut1:cut2,0][:gap_at+1]
            lcaftergap  = lc[cut1:cut2,0][gap_at+1:]
            if len(lcbeforegap) < len(lcaftergap)*0.2:
                # if gap is NOT in the middle of flare -> recalculate flare position (skip before lc)
                try:
                    cut1 = um[ii] + gap_at+1
                    cut2 = um[ii+1]
                    dt = lc[cut2,0]-lc[cut1,0]
                except IndexError:
                    cut1 = um[ii] + gap_at+1
                    cut2 = -1
                    dt = lc[cut2,0]-lc[cut1,0]
            elif len(lcaftergap) < len(lcbeforegap)*0.2:
                # if gap is NOT in the middle of flare -> recalculate flare position (skip after lc)
                try:
                    cut1 = um[ii]
                    cut2 = um[ii] + gap_at
                    dt = lc[cut2,0]-lc[cut1,0]
                except IndexError:
                    cut1 = um[ii]
                    cut2 = -1
                    dt = lc[cut2,0]-lc[cut1,0]
            else:
                if debug: print('Skipping flare has a large gap')
                # if flare has a large gap -> remove because it is a FP
                continue

        if not inputdata and len(lc[cut1:cut2,0]) < dt/np.median(np.diff(lc[:,0]))*0.4 and dt<1/24:
            # if duty cycle in short flare is < 50%
            if debug: print('Skipping short flare duty cycle in flare is < 40%')
            continue

        if not inputdata and len(lc[cut1:cut2,0]) < dt/np.median(np.diff(lc[:,0]))*0.2 and dt>=0.1  :
            # if duty cycle in long flare is < 20%
            if debug: print('Skipping long flare duty cycle in flare is <20 %')
            continue

        if not inputdata and len(lc[cut1:cut2,0]) < 3:
            # 3 pts is not a flare
            if debug: print('Skipping flare < 4 pts')
            continue

        # ---- Define flare, fitting and plottiong window lengths ----------------
        umshort = (lc[:,0]>=lc[cut1,0]) & (lc[:,0]<=lc[cut2,0])
        umwide = (lc[:,0]>=lc[cut1,0]-dt) & (lc[:,0]<=lc[cut2,0]+dt)
        umplot = (lc[:,0]>=lc[cut1,0]-2*dt) & (lc[:,0]<=lc[cut2,0]+2*dt)
        umbefore = (lc[:,0]>=lc[cut1,0]-dt) & (lc[:,0]<=lc[cut2,0])
        umafter = (lc[:,0]>=lc[cut1,0]) & (lc[:,0]<=lc[cut2,0]+dt)

        if stella:
            # The whole flare needed, not just the peak!
            umshort = (lc[:,0]>=lc[cut1,0]-2*dt) & (lc[:,0]<=lc[cut2,0]+2*dt)
            umwide = (lc[:,0]>=lc[cut1,0]-4*dt) & (lc[:,0]<=lc[cut2,0]+4*dt)
            umplot = (lc[:,0]>=lc[cut1,0]-6*dt) & (lc[:,0]<=lc[cut2,0]+6*dt)

        # ---- Fit 2nd order polynomial iteratively (to get bkg) + model flare ----------------
        try:
            res,convy,linfit,linfitshort,tpeak0max,fwhm0,amp0,z = fit_bkg_and_flare_model(lc,umshort,umwide,order=2,debug=debug)
        except:
            continue

        if plotting:
            fig = plt.figure(figsize=(15,8))
            plt.plot(lc[umplot,0], lc[umplot,1],'k.',ms=10)
            plt.plot(lc[umplot,0], lc[umplot,1],'lightgray')
            plt.plot(lc[umwide,0],       lc[umwide,1], c='k')        ###"""umshort"""
            plt.plot(lc[umwide,0],       convy + linfitshort, 'rx')        ###"""umshort"""
            plt.plot(lc[umwide,0], linfit, c='C1',label='Initial polynomial')

        # ---- Remove flare to measure bkg variation and get flare S/N ----------------
        residuallc = lc[umshort,1] - (np.poly1d(res.x[:-3])(lc[umshort,0]) + aflare1_fast(lc[umshort,0],*res.x[-3:]))
        residuallcSperN = res.x[-1]/np.std(residuallc,ddof=6)
        residuallcwide = lc[umwide,1] - (np.poly1d(res.x[:-3])(lc[umwide,0]) + aflare1_fast(lc[umwide,0],*res.x[-3:]))
        residuallcwideSperN = res.x[-1]/np.std(residuallcwide,ddof=6)
        residuallcbefore = lc[umbefore,1] - (np.poly1d(res.x[:-3])(lc[umbefore,0]) + aflare1_fast(lc[umbefore,0],*res.x[-3:]))
        residuallcbeforeSperN = res.x[-1]/np.std(residuallcbefore,ddof=6)
        residuallcafter = lc[umafter,1] - (np.poly1d(res.x[:-3])(lc[umafter,0]) + aflare1_fast(lc[umafter,0],*res.x[-3:]))
        residuallcafterSperN = res.x[-1]/np.std(residuallcafter,ddof=6)

        # ---- There is another large flare in the wide (3x) window -> fit only in smaller window ----------------
        if residuallcwideSperN < residuallcSperN/1.8:
            if debug:
                print('Smaller window needed! Large flare nearby!')
                print('S/N=', residuallcSperN )
                print('S/N wide=',residuallcwideSperN  )
                print('S/N before=',residuallcbeforeSperN  )
                print('S/N after=',residuallcafterSperN  )
            if residuallcbeforeSperN <= residuallcSperN/1.8 and residuallcafterSperN <= residuallcSperN/1.8:
                # both side gives worse S/N -> use only small window
                umsmaller = umshort
                if debug: print('Using small window!')
            elif residuallcbeforeSperN <= residuallcSperN/1.8:
                # left side gives worse S/N-> use only after window
                umsmaller = umafter
                if debug: print('Using after window!')
            elif residuallcafterSperN <= residuallcSperN/1.8:
                # right side gives worse S/N-> use only before window
                if debug: print('Using before window!')
                umsmaller = umbefore
            else:
                # probably only the middle part of the flare is labeled
                if debug: print('Using plotting size window!')
                umsmaller = umplot

            # ---- Get lower bounds of poly+flare model fitting ----------------
            lowbound = []
            for coefficient in z[:-1]:
                lowbound.append( coefficient-np.abs(coefficient)*0.002  )
            lowbound = tuple(lowbound)
            lowbound += (z[-1]-np.abs(z[-1])*0.0001,  lc[umwide,0].min(),2/(24*60),0 )

            # ---- Get upper bounds of poly+flare model fitting ----------------
            upbound = []
            for coefficient in z:
                upbound.append( coefficient+np.abs(coefficient)*0.002  )
            upbound = tuple(upbound)
            upbound += ( lc[umwide,0].max(),np.inf,10*amp0  )

            # ---- Fit polynomial and bkg at once ----------------
            res = least_squares(diff_linear_nopeak2, [*z,tpeak0max,fwhm0,amp0],
                                bounds=( lowbound , upbound ),
                                args=(lc[umsmaller,0], lc[umsmaller,1]))

            # ---- Remove flare to measure bkg variation and get flare S/N ----------------
            residuallc = lc[umshort,1] - (np.poly1d(res.x[:-3])(lc[umshort,0]) + aflare1_fast(lc[umshort,0],*res.x[-3:]))
            residuallcSperN = res.x[-1]/np.std(residuallc,ddof=6)
            residuallcwide = lc[umsmaller,1] - (np.poly1d(res.x[:-3])(lc[umsmaller,0]) + aflare1_fast(lc[umsmaller,0],*res.x[-3:]))
            residuallcwideSperN = res.x[-1]/np.std(residuallcwide,ddof=6)
            umwide = umsmaller

        # ---- Fit tries to follow large variation like pulsation/eclipses ----------------
        poly_first_point = np.poly1d(res.x[:-3])(lc[umshort,0][0])
        poly_last_point  = np.poly1d(res.x[:-3])(lc[umshort,0][-1])
        if lc[umshort,1][0]-5*np.std(residuallc,ddof=6) > poly_first_point or \
            lc[umshort,1][0]+5*np.std(residuallc,ddof=6) < poly_first_point or \
            lc[umshort,1][-1]-5*np.std(residuallc,ddof=6) > poly_last_point or \
            lc[umshort,1][-1]+5*np.std(residuallc,ddof=6) < poly_last_point:

            if debug:
                print('Wrong fit! Smaller window needed!')
            umsmaller = umshort

            # ---- Fit 2nd order polynomial iteratively (to get bkg) + model flare ----------------
            try:
                res,convy,linfit,linfitshort,tpeak0max,fwhm0,amp0,z = fit_bkg_and_flare_model(lc,umshort,umshort,order=2,debug=debug)
            except:
                continue

            # ---- Get lower bounds of poly+flare model fitting ----------------
            lowbound = []
            for coefficient in z[:-1]:
                lowbound.append( coefficient-np.abs(coefficient)*0.002  )
            lowbound = tuple(lowbound)
            lowbound += (z[-1]-np.abs(z[-1])*0.0001,  lc[umsmaller,0].min(),2/(24*60),0 )

            # ---- Get upper bounds of poly+flare model fitting ----------------
            upbound = []
            for coefficient in z:
                upbound.append( coefficient+np.abs(coefficient)*0.002  )
            upbound = tuple(upbound)
            upbound += ( lc[umsmaller,0].max(),np.inf,10*amp0  )

            # ---- Fit polynomial and bkg at once ----------------
            res = least_squares(diff_linear_nopeak2, [*z,tpeak0max,fwhm0,amp0],
                                bounds=( lowbound , upbound ),
                                args=(lc[umsmaller,0], lc[umsmaller,1]))

            # ---- Remove flare to measure bkg variation and get flare S/N ----------------
            residuallc = lc[umshort,1] - (np.poly1d(res.x[:-3])(lc[umshort,0]) + aflare1_fast(lc[umshort,0],*res.x[-3:]))
            residuallcSperN = res.x[-1]/np.std(residuallc,ddof=6)
            residuallcwide = lc[umsmaller,1] - (np.poly1d(res.x[:-3])(lc[umsmaller,0]) + aflare1_fast(lc[umsmaller,0],*res.x[-3:]))
            residuallcwideSperN = res.x[-1]/np.std(residuallcwide,ddof=6)
            umwide = umsmaller

        # ---- Calculate Bayesian Information Criterion for 2nd order polyfit + flare model ------
        chi2 = (np.power( residuallcwide, 2 ) ).sum()
        ddf = 2+3 # 2nd order poly + flare model
        BIC2 = get_BIC(chi2,len(residuallcwide),ddf)
        if debug:
            print('BIC is %.6f for order %d' % (BIC2,2))
            print('Chi2 is %.6f for order %d' % (chi2,2))

        # ---- Fit 3nd order polynomial iteratively (to get bkg) + model flare ----------------
        try:
            res3,convy3,linfit3,linfitshort3,tpeak0max3,fwhm03,amp03,z = fit_bkg_and_flare_model(lc,umshort,umwide,order=3,debug=debug)
        except:
            continue

        # ---- Remove flare to measure bkg variation and get flare S/N ----------------
        residuallc3 = lc[umshort,1] - (np.poly1d(res3.x[:-3])(lc[umshort,0]) + aflare1_fast(lc[umshort,0],*res3.x[-3:]))
        residuallcSperN3 = res3.x[-1]/np.std(residuallc3,ddof=6)
        residuallcwide3 = lc[umwide,1] - (np.poly1d(res3.x[:-3])(lc[umwide,0]) + aflare1_fast(lc[umwide,0],*res3.x[-3:]))
        residuallcwideSperN3 = res3.x[-1]/np.std(residuallcwide3,ddof=6)

        # ---- Calculate Bayesian Information Criterion for 3rd order polyfit + flare model ------
        chi2 = (np.power( residuallcwide3, 2 ) ).sum()
        ddf = 3+3 # 3nd order poly + flare model
        BIC3 = get_BIC(chi2,len(residuallcwide3),ddf)
        if debug:
            print('BIC is %.6f for order %d' % (BIC3,3))
            print('Chi2 is %.6f for order %d' % (chi2,3))

        # ----- Check if 3rd order polyfit is better than 2nd order -------
        # ----- if true -> use 3rd order parameters instead ---------------
        if BIC3 < BIC2:
            if debug: print('USING 3rd ORDER POLYFIT!')
            res,convy,linfit,linfitshort = res3,convy3,linfit3,linfitshort3
            residuallc,residuallcSperN = residuallc3,residuallcSperN3
            residuallcwide,residuallcwideSperN = residuallcwide3,residuallcwideSperN3
            tpeak0max,fwhm0,amp0 = tpeak0max3,fwhm03,amp03

            # ---- Fit 5th order polynomial iteratively (to get bkg) + model flare ----------------
            try:
                res5,convy5,linfit5,linfitshort5,tpeak0max5,fwhm05,amp05,z = fit_bkg_and_flare_model(lc,umshort,umwide,order=5,debug=debug)
            except:
                continue

            # ---- Remove flare to measure bkg variation and get flare S/N ----------------
            residuallc5 = lc[umshort,1] - (np.poly1d(res5.x[:-3])(lc[umshort,0]) + aflare1_fast(lc[umshort,0],*res5.x[-3:]))
            residuallcSperN5 = res3.x[-1]/np.std(residuallc5,ddof=6)
            residuallcwide5 = lc[umwide,1] - (np.poly1d(res5.x[:-3])(lc[umwide,0]) + aflare1_fast(lc[umwide,0],*res5.x[-3:]))
            residuallcwideSperN5 = res5.x[-1]/np.std(residuallcwide5,ddof=6)

            # ---- Calculate Bayesian Information Criterion for 5rd order polyfit + flare model ------
            chi2 = (np.power( residuallcwide5, 2 ) ).sum()
            ddf = 5+3 # 5th order poly + flare model
            BIC5 = get_BIC(chi2,len(residuallcwide5),ddf)
            if debug:
                print('BIC is %.6f for order %d' % (BIC5,5))
                print('Chi2 is %.6f for order %d' % (chi2,5))

            # ----- Check if 5th order polyfit is better than 2nd and 3rd order -------
            # ----- if true -> use 5th order parameters instead ---------------
            if BIC5 < BIC2 and BIC5 < BIC3:
                if debug: print('USING 5th ORDER POLYFIT!')
                res,convy,linfit,linfitshort = res5,convy5,linfit5,linfitshort5
                residuallc,residuallcSperN = residuallc5,residuallcSperN5
                residuallcwide,residuallcwideSperN = residuallcwide5,residuallcwideSperN5
                tpeak0max,fwhm0,amp0 = tpeak0max5,fwhm05,amp05

        # ---- Get final bkg polynomial ----------------
        linfit = np.poly1d(res.x[:-3])(lc[umwide,0])

        if plotting: plt.plot(lc[umwide,0], linfit, c='C2',label='LSQ polynomial')

        # ---- Generate flare model curve ----------------
        tplotx = np.linspace(lc[umwide,0].min(),lc[umwide,0].max(),1000)
        if plotting: plt.plot( tplotx,  np.poly1d(res.x[:-3])(tplotx) + aflare1_fast(tplotx,*res.x[-3:]),lw=2,c='r',label='Flare+Polynomial' )
        modellllll = aflare1_fast(tplotx,*res.x[-3:])
        modellum = modellllll > np.std(residuallcwide,ddof=6)
        modellum += (tplotx<res.x[-3]) & (modellllll>0)
        try:
            mainflarestartsat = tplotx[ modellum ].min()
            mainflareendsat   = tplotx[ modellum ].max()
        except ValueError:
            # flare peak is below 1 sigma -> drop it
            if debug: print('Flare peak is below 1 sigma -- Skipping')
            plt.close()
            continue

        # ---- Collect S/N --------
        if debug:
            print('S/N=', residuallcSperN )
            print('S/N wide=',residuallcwideSperN  )
        SperN_all.append(residuallcSperN)            ####"""residuallcwideSperN"""

        # ---- Collect flare parameters ----------------
        if plotting:
            plt.fill_between(tplotx[ modellum ],modellllll[ modellum ] + np.poly1d(res.x[:-3])(tplotx)[modellum ],np.poly1d(res.x[:-3])(tplotx)[modellum ],alpha=0.3,color='b')
            plt.plot(tplotx[ modellum ],modellllll[ modellum ] + np.poly1d(res.x[:-3])(tplotx)[modellum ],'m',lw=3,ls='dashed')
        try:
            flare_epsilon = simps(modellllll[ modellum ],x=tplotx[ modellum ])
            flare_duration = tplotx[ modellum ].ptp()
        except IndexError:
            flare_epsilon = flare_duration = np.nan
        if debug: print('Epsilon 1sigma lc=', flare_epsilon)
        if debug: print('duration=', flare_duration,'days')
        #if residuallcwideSperN>3:
        flare_duration_all.append(flare_duration*u.day)
        flare_duration_main = flare_duration
        modeltpeak_all.append(res.x[-3])
        modelamp_all.append(res.x[-1])
        convy = median_filter(lc[umshort,1]-np.poly1d(res.x[:-3])(lc[umshort,0]) ,3, mode='mirror' )
        tpeak_max_final = lc[umshort,0][ np.argmax(convy) ]
        tpeak_all.append(tpeak_max_final)
        amp_all.append(np.max(convy))

        if debug:
            print('Tpeak0max=',tpeak0max,'->',tpeak_max_final)
            print('FHWM0=',fwhm0,'->',res.x[-2])
            print('Amp0=',amp0,'->',res.x[-1])

        # ---- Integrate flare model energy above 1 sigma std / save and plot it  ----------------
        sigma1 = np.std(residuallcwide,ddof=6)
        sigma1pts = np.where( aflare1_fast(lc[umwide,0],*res.x[-3:]) >= sigma1 )
        sigma1pts = ( lc[umwide,1][ sigma1pts ] -linfit[ sigma1pts ] ) / sigma1
        #print( sigma1pts )
        if (np.diff(np.where(sigma1pts>=3)) == 1).sum() >= 2:
            if debug: print('---> Significant!')
        if res.x[-1]/np.std(residuallcwide,ddof=6) >= 3 and plotting:
            ax = plt.gca()
            ax.set_facecolor('xkcd:mint')
        elif res.x[-1]/np.std(residuallcwide,ddof=6) <= 2 and plotting:
            ax = plt.gca()
            ax.set_facecolor('xkcd:salmon pink')

        if plotting:
            plt.plot(lc[umwide,0],sigma1 +linfit,c='b',alpha=0.7,zorder=0, label='1-3 $\sigma$ limits')
            plt.plot(lc[umwide,0],3*sigma1 +linfit,c='b',alpha=0.7,zorder=0)
            plt.axvline(tpeak0max,c='C0',zorder=0,alpha=0.5)

            plt.ylim(lc[umplot,1].min()-lc[umplot,1].ptp()*0.2,lc[umplot,1].max()+lc[umplot,1].ptp()*0.2)
            plt.xlim(lc[umplot,0].min()-0.004,lc[umplot,0].max()+0.004)
            plt.legend()
            plt.show()
            plt.close(fig)

        try:
            flare_epsilon = simps(modellllll[ modellum ]/np.poly1d(res.x[:-3])(tplotx[ modellum ]),x=tplotx[ modellum ])
        except IndexError:
            flare_epsilon = np.nan
        if debug: print('Flare E=',flare_epsilon )
        flare_epsilon_all.append(flare_epsilon)

        # ---- Check if there are other flares in the window ----------------
        secondaryflare_sigma = secondaryflare_sigma
        sigma1 = np.std(residuallc)
        sigma1wide = np.std(residuallcwide)

        if plotting:
            fig = plt.figure(figsize=(15,4))
            plt.plot(lc[umwide,0],residuallcwide,'k.',label='Residual',ls='-')
            plt.axhline(secondaryflare_sigma*sigma1wide,c='r',label='%.1f $\sigma$ limit' % secondaryflare_sigma)
            if np.any(np.diff(np.where(np.diff( np.where( residuallc >= secondaryflare_sigma*sigma1wide )[0] )==1)[0])==1):
                if debug: print('We have another flare in the window!')
                ax = plt.gca()
                ax.set_facecolor('xkcd:sky blue')

        groups = np.where( residuallcwide >= secondaryflare_sigma*sigma1wide )[0]
        groups = np.split(groups, np.where(np.diff(groups)>1)[0]+1)
        if plotting:
            for group in groups:
                if len(group)>2:
                    #print(group)
                    plt.plot(lc[umwide,0][group],residuallcwide[group],'rx')
            plt.axhline(0,c='gray',zorder=0,alpha=0.5)
            plt.xlim(lc[umplot,0].min()-0.004,lc[umplot,0].max()+0.004)
            plt.legend()
            plt.show()
            plt.close(fig)

        # ---- Fit and remove other flares iteratively ----------------
        # ---- Must have more than 3 consecutive pts w/ S/N > secondaryflare_sigma * bkg std ----------------
        secondary_flare_params = []
        nsecondaryflares = 0
        while True:
            if np.any(np.diff(np.where(np.diff( np.where( residuallcwide >= secondaryflare_sigma*sigma1wide )[0] )==1)[0])==1):
                # ---- Group flares by different consecutive pts ----------------
                groups = np.where( residuallcwide >= secondaryflare_sigma*sigma1wide )[0]
                groups = np.split(groups, np.where(np.diff(groups)>1)[0]+1)
                for group in groups:
                    if len(group)<3:
                        residuallcwide[group] = np.mean(residuallcwide)

                # ---- Remove bkg and estimate initial flare parameters ----------------
                # Elliminate 1 pt outliers before finding maximum
                convy = median_filter(residuallcwide,3, mode='mirror')

                down = np.percentile(residuallcwide,5)
                amp0 = np.ptp(residuallcwide[residuallcwide>=down])
                if amp0==0:
                    amp0 = np.ptp(residuallcwide)
                fwhm0 = 5./60./24.
                tpeak0max = lc[umwide,0][ np.argmax(convy) ]

                # ---- Fit flare model to residual lc ----------------
                res2 = least_squares(diff, [tpeak0max,fwhm0,amp0],
                                bounds=( (lc[umwide,0].min(),2/(24*60),0) , (lc[umwide,0].max(),np.inf,10*amp0) ),
                                args=(lc[umwide,0], residuallcwide)
                                    )

                if plotting:
                    fig = plt.figure(figsize=(15,4))
                    plt.plot(lc[umwide,0], residuallcwide,'k.',ms=10)
                    plt.plot(lc[umwide,0], residuallcwide,'lightgray')
                    plt.axvline(tpeak0max,c='r',alpha=0.5,zorder=0)
                    plt.plot( lc[umwide,0], aflare1_fast(lc[umwide,0],*res2.x),lw=2,c='r',label='Residual Flare+Polynomial' )

                # ---- Get residual after flare fitting ----------------
                residuallc = residuallc -  aflare1_fast(lc[umshort,0],*res2.x)
                if debug: print('S/N=', res2.x[-1]/sigma1 )
                residuallcwide = residuallcwide -  aflare1_fast(lc[umwide,0],*res2.x)

                if debug:
                    print('S/N wide=', res2.x[-1]/sigma1wide )
                    print('tmax',res2.x[0])
                if res2.x[-1]/sigma1wide < 0.5:
                    if plotting:
                        plt.close(fig)
                    break

                # ---- Get model above 1 sigma std for integration to get epsilon ----------------
                tplotx = np.linspace(lc[umwide,0].min(),lc[umwide,0].max(),1000)
                modellllll = aflare1_fast(tplotx,*res2.x)
                modellum = modellllll > sigma1wide
                modellum += (tplotx<res2.x[0]) & (modellllll>0)

                # ---- Save flare if S/N > secondaryflare_sigma * bgk std ----------------
                if res2.x[-1]/sigma1wide>secondaryflare_sigma:
                    try:
                        # ---- Integrate flare model energy above 1 sigma std / save it  ----------------
                        cut_to_fitted_region = aflare1_fast(lc[umwide,0],*res2.x) > sigma1wide
                        tpeak_max_final = lc[umwide,0][cut_to_fitted_region][ np.argmax(convy[cut_to_fitted_region]) ]
                        flare_epsilon = simps(modellllll[ modellum ]/np.poly1d(res.x[:3])(tplotx[ modellum ]),x=tplotx[ modellum ])
                        flare_duration = tplotx[ modellum ].ptp()
                        if res2.x[0]<mainflarestartsat or mainflareendsat<res2.x[0]:
                            # Save side peaks only if they are not too close to main peak
                            # i.e. they are outside of flare duration window
                            if debug: print('xxxxx Flare saved!  xxxxx')
                            flare_duration_all.append(flare_duration*u.day)
                            flare_epsilon_all.append(flare_epsilon)
                            modeltpeak_all.append(res2.x[0])
                            modelamp_all.append(res2.x[2])
                            convy = median_filter(residuallcwide, 3 , mode='mirror')
                            tpeak_all.append(tpeak_max_final)
                            amp_all.append(np.max(convy))
                            SperN_all.append(res2.x[-1]/sigma1wide)

                        # Collect secondary flare params to be fitted simultaneously
                        secondary_flare_params.append(res2.x)
                        nsecondaryflares += 1
                    except:
                        # Peak is out of flagged region
                        if plotting:
                            plt.close(fig)
                        continue

                if plotting:
                    plt.fill_between(tplotx[ modellum ],modellllll[ modellum ] ,alpha=0.3,color='b')
                    plt.axhline(secondaryflare_sigma*sigma1wide,c='r',ls='--',label='%.1f $\sigma$ limit' % secondaryflare_sigma)
                    plt.axhline(0,c='gray',zorder=0,alpha=0.5)
                    plt.legend()
                    plt.xlim(lc[umwide,0].min()-0.004,lc[umwide,0].max()+0.004)
                    plt.show()
                    plt.close(fig)

            else:
                break

        # --- Fit bkg and all identified flares simultaneously ---
        # --- Then search for missed flares in the wide window --
        if nsecondaryflares > 0:
            # Add original large flare to flare list
            nsecondaryflares += 1

            # Collect all flare params
            allflareparam = res.x.tolist()
            for sfp in secondary_flare_params:
                allflareparam += sfp.tolist()

            # --- Fitting bkg + all flares ---
            resmulti = fit_bkg_and_multi_flare_model(lc,umwide,
                                                     allflareparam,
                                                     nflares=nsecondaryflares,
                                                     debug=debug)

            if plotting:
                # --- Plot results ---
                fig = plt.figure(figsize=(15,4))
                plt.plot(lc[umplot,0], lc[umplot,1],'k',ms=10)
                plt.scatter(lc[umwide,0], lc[umwide,1], c='C0',zorder=5,s=3)

                # bkg + main peak + side peak fitted separately
                fitwide = np.poly1d(allflareparam[:-(nsecondaryflares*3)]) (lc[umwide,0])
                for fl in reversed(range(0,nsecondaryflares*3,3 )):
                    fitwide += aflare1_fast(lc[umwide,0], allflareparam[-(fl+3)],allflareparam[-(fl+2)],allflareparam[-(fl+1)])
                plt.plot(lc[umwide,0], fitwide,lw=2,c='k',ls='--',zorder=10,label='Separate fits')
                # bkg + main peak + side peak fitted simultaneously
                fitwide = np.poly1d(resmulti.x[:-(nsecondaryflares*3)]) (lc[umwide,0])
                for fl in reversed(range(0,nsecondaryflares*3,3 )):
                    fitwide += aflare1_fast(lc[umwide,0], resmulti.x[-(fl+3)],resmulti.x[-(fl+2)],resmulti.x[-(fl+1)])
                plt.plot(lc[umwide,0], fitwide,lw=2,c='r',zorder=15,label='Simultaneous fit')

                plt.legend()
                plt.show()
                plt.close(fig)

                del fitwide

            # --- Get residual lc after fitting bkg and all flare components ---
            residuallcwide = np.poly1d(resmulti.x[:-(nsecondaryflares*3)]) (lc[umwide,0])
            for fl in reversed(range(0,nsecondaryflares*3,3 )):
                residuallcwide += aflare1_fast(lc[umwide,0], resmulti.x[-(fl+3)],resmulti.x[-(fl+2)],resmulti.x[-(fl+1)])

            residuallcwide = lc[umwide,1] - residuallcwide

            # --- Get residual lc after fitting bkg and all flare componentsfor short duration ---
            residuallc = np.poly1d(resmulti.x[:-(nsecondaryflares*3)]) (lc[umshort,0])
            for fl in reversed(range(0,nsecondaryflares*3,3 )):
                residuallc += aflare1_fast(lc[umshort,0], resmulti.x[-(fl+3)],resmulti.x[-(fl+2)],resmulti.x[-(fl+1)])

            residuallc = lc[umshort,1] - residuallc

            # --- Fit polynomial to residual lc to remove trends ---
            with warnings.catch_warnings(record=True):
                polwide = np.poly1d(np.polyfit(lc[umwide,0],residuallcwide,31))
            pol =     polwide(lc[umshort,0])
            polwide = polwide(lc[umwide,0])

            if plotting:
                # --- Plot results ---
                fig = plt.figure(figsize=(15,4))
                plt.plot(lc[umwide,0], residuallcwide,'k',ms=10,label='Residual lc w/out flares')
                plt.scatter(lc[umwide,0], residuallcwide, c='C0',zorder=5,s=3)
                plt.plot(lc[umwide,0],polwide,'r',label='Trend to be removed')
                plt.legend()
                plt.show()
                plt.close(fig)

            # --- remove additional trends ---
            residuallcwide -= polwide
            residuallc     -= pol

            # ---- Fit and remove other flares iteratively ----------------
            # ---- Must have more than 3 consecutive pts w/ S/N > secondaryflare_sigma * bkg std ----------------
            sigma1wide = np.std(residuallcwide)
            while True:
                if np.any(np.diff(np.where(np.diff( np.where( residuallcwide >= secondaryflare_sigma*sigma1wide )[0] )==1)[0])==1):
                    # ---- Group flares by different consecutive pts ----------------
                    groups = np.where( residuallcwide >= secondaryflare_sigma*sigma1wide )[0]
                    groups = np.split(groups, np.where(np.diff(groups)>1)[0]+1)
                    for group in groups:
                        if len(group)<3:
                            residuallcwide[group] = np.mean(residuallcwide)

                    # ---- Remove bkg and estimate initial flare parameters ----------------
                    # Elliminate 1 pt outliers before finding maximum
                    convy = median_filter(residuallcwide,3, mode='mirror')

                    down = np.percentile(residuallcwide,5)
                    amp0 = np.ptp(residuallcwide[residuallcwide>=down])
                    if amp0==0:
                        amp0 = np.ptp(residuallcwide)
                    fwhm0 = 5./60./24.
                    tpeak0max = lc[umwide,0][ np.argmax(convy) ]

                    # ---- Fit flare model to residual lc ----------------
                    res2 = least_squares(diff, [tpeak0max,fwhm0,amp0],
                                    bounds=( (lc[umwide,0].min(),2/(24*60),0) , (lc[umwide,0].max(),np.inf,10*amp0) ),
                                    args=(lc[umwide,0], residuallcwide)
                                        )

                    if plotting:
                        fig = plt.figure(figsize=(15,4))
                        plt.plot(lc[umwide,0], residuallcwide,'k.',ms=10)
                        plt.plot(lc[umwide,0], residuallcwide,'lightgray')
                        #plt.plot(lc[umshort,0],       residuallc, c='lightgray')
                        plt.axvline(tpeak0max,c='r',alpha=0.5,zorder=0)
                        plt.plot( lc[umwide,0], aflare1_fast(lc[umwide,0],*res2.x),lw=2,c='r',label='Residual Flare+Polynomial' )

                    # ---- Get residual after flare fitting ----------------
                    residuallc = residuallc -  aflare1_fast(lc[umshort,0],*res2.x)
                    if debug: print('S/N=', res2.x[-1]/sigma1 )
                    residuallcwide = residuallcwide -  aflare1_fast(lc[umwide,0],*res2.x)

                    if debug:
                        print('S/N wide=', res2.x[-1]/sigma1wide )
                        print('tmax',res2.x[0])
                    if res2.x[-1]/sigma1wide < 0.5:
                        if plotting:
                            plt.close(fig)
                        break

                    # ---- Get model above 1 sigma std for integration to get epsilon ----------------
                    tplotx = np.linspace(lc[umwide,0].min(),lc[umwide,0].max(),1000)
                    modellllll = aflare1_fast(tplotx,*res2.x)
                    modellum = modellllll > sigma1wide
                    modellum += (tplotx<res2.x[0]) & (modellllll>0)

                    # ---- Save flare if S/N > secondaryflare_sigma * bgk std ----------------
                    if res2.x[-1]/sigma1wide>secondaryflare_sigma:
                        try:
                            # ---- Integrate flare model energy above 1 sigma std / save it  ----------------
                            cut_to_fitted_region = aflare1_fast(lc[umwide,0],*res2.x) > sigma1wide
                            tpeak_max_final = lc[umwide,0][cut_to_fitted_region][ np.argmax(convy[cut_to_fitted_region]) ]
                            flare_epsilon = simps(modellllll[ modellum ]/np.poly1d(res.x[:3])(tplotx[ modellum ]),x=tplotx[ modellum ])
                            flare_duration = tplotx[ modellum ].ptp()
                            if res2.x[0]<mainflarestartsat or mainflareendsat<res2.x[0]:
                                # Save side peaks only if they are not too close to main peak
                                # i.e. they are outside of flare duration window
                                if debug: print('xxxxx Flare saved!  xxxxx')
                                flare_duration_all.append(flare_duration*u.day)
                                flare_epsilon_all.append(flare_epsilon)
                                modeltpeak_all.append(res2.x[0])
                                modelamp_all.append(res2.x[2])
                                convy = median_filter(residuallcwide, 3 , mode='mirror')
                                tpeak_all.append(tpeak_max_final)
                                amp_all.append(np.max(convy))
                                SperN_all.append(res2.x[-1]/sigma1wide)
                        except:
                            # Peak is out of flagged region
                            if plotting:
                                plt.close(fig)
                            continue

                    if plotting:
                        plt.fill_between(tplotx[ modellum ],modellllll[ modellum ] ,alpha=0.3,color='b')
                        plt.axhline(secondaryflare_sigma*sigma1wide,c='r',ls='--',label='%.1f $\sigma$ limit' % secondaryflare_sigma)
                        plt.axhline(0,c='gray',zorder=0,alpha=0.5)
                        plt.legend()
                        plt.xlim(lc[umwide,0].min()-0.004,lc[umwide,0].max()+0.004)
                        plt.show()
                        plt.close(fig)

                else:
                    break

    return modeltpeak_all,modelamp_all,tpeak_all,amp_all,SperN_all,flare_epsilon_all,flare_duration_all

def get_medfilted_lc(lc,lcorig,window_size=64):
    """
    Median filter ipred lc then return prediction
    with the orignal sampling.
    """
    time_sort, pred_sort = lc.T[0], lc.T[2]

    mfilt=np.array(medfilt(pred_sort, kernel_size=window_size-1))

    time, flux = lcorig.T[0], lcorig.T[1]

    imfilt = interp1d(time_sort, mfilt, bounds_error=False, fill_value=np.nan)

    return imfilt(time)

def validation(time,flux,ipred, window_size=64, pred_probability_cut=0.5,
                progressbar=True, debug=False, plotting=False,
                secondaryflare_sigma=1.5):
    """
    This function uses the original light curve and the uniformly sampled
    LSTM prediction to fit each flare above a given threshold.

    Parameters
    ----------
    time : array
        Time values of the original light curve.
    flux : array
        Flux values of the original light curve.
    ipred : ndarray
        The light curve and predictions interpolated to a uniformly sampled
        grid given by `flatwrm2.prediction`.
    window_size : int, default: 64
        The window size of median filter kernel used to smooth the raw LSTM
        predictions. Must be even!
        Note: actually window_size-1 will be used.
    pred_probability_cut : float, default: 0.5
        The probability threshold above which a point is consider to be a
        flare after the median filter smoothing.
    debug : boolen, default: False
        If `True`, messages that are useful for debugging will be printed.
    plotting : boolen, default: False
        If `True`, plots that are useful for debugging will be displayed.
    secondaryflare_sigma : float, default: 1.5
        If multiple flares are found in a given flare window, then other
        flares than the one with the largest amplitude will be accepted
        if they have S/N > `secondaryflare_sigma`.

    Returns
    -------
    flare_parameters : pandas DataFrame
        Parameters of detected and fitted flares.

        t_peak: peak time of maximum point.
        Amplitude: amplitude of maximum point.
        t_peak_model: peak time of flare model.
        Amplitude_model: amplitude of flare model.
        SperN: flare signal-to-noise ratio.
        Epsilon: integrated flare flux in relative units.
        Duration: flare duration in days.
    """

    # Concat input lc
    lc = np.c_[time,flux]

    # --- Create array for reinterpolated LSTM predictions ---
    lclstm = np.empty( (lc.shape[0],3) )
    lclstm[:,0] = lc[:,0]
    lclstm[:,1] = lc[:,1]

    # --- Apply medfiltering with desired window size ---
    lclstm[:,2] = get_medfilted_lc(ipred,lc,window_size=window_size)

    # ---- Drop useless points -----
    goodpts = np.isfinite(lclstm[:,1])
    lclstm  = lclstm[goodpts]

    # ---- Divide lc by its median to get relative flare amplitudes -----
    lclstm[:,1] /= np.nanmedian(lclstm[:,1])

    # ---- Run flare fitting algorithm -----
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        (modeltpeak,modelamp,tpeak,amp,SperN,
         flare_epsilon,flare_duration) = get_flares(lclstm,
                                             plotting=plotting,
                                             debug=debug,
                                             secondaryflare_sigma=secondaryflare_sigma,
                                             pred_probability_cut=pred_probability_cut,
                                             progressbar=progressbar)

    # ---- Collect fitted flare parameters  -----
    _,uniquekey = np.unique(tpeak,return_index=True)

    modeltpeakorig_lstm = np.array(modeltpeak)[uniquekey]
    modelamp_lstm       = np.array(modelamp)[uniquekey]
    tpeakorig_lstm      = np.array(tpeak)[uniquekey]
    amp_lstm            = np.array(amp)[uniquekey]
    SperN_lstm          = np.array(SperN)[uniquekey]
    flare_epsilon_lstm  = np.array(flare_epsilon)[uniquekey]
    flare_duration_lstm = [x.value for index,x in enumerate(flare_duration) if index in uniquekey]

    # --- Store results ---
    flare_parameters = {}
    flare_parameters['t_peak']          = tpeakorig_lstm
    flare_parameters['Amplitude']       = amp_lstm
    flare_parameters['t_peak_model']    = modeltpeakorig_lstm
    flare_parameters['Amplitude_model'] = modelamp_lstm
    flare_parameters['SperN']           = SperN_lstm
    flare_parameters['Epsilon']         = flare_epsilon_lstm
    flare_parameters['Duration']        = flare_duration_lstm

    flare_parameters = pd.DataFrame(flare_parameters)

    return flare_parameters
