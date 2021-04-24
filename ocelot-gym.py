#!/usr/bin/env python3
# coding: utf-8

#from matplotlib import pyplot as plt
import numpy as np

from aflare import *
from pymacula import *


tmin = 0.1
tmax= 100. #days
cadence = 5. /60./24 #min
t = np.arange(tmin, tmax, cadence )

Nstar = 100
periods = np.random.lognormal(mean=np.log(5), sigma=0.5, size=Nstar)
if periods.min() < 0.4:
    periods += 0.4 - periods.min()
    
for star in np.arange(Nstar):
    theta_star = np.zeros(12)
    theta_star[0] = np.pi/2. #inclination
    theta_star[1] = periods[star] #Prot
    theta_star[2:4] = np.array([0.2, 0.2])
    theta_star[4:8] = np.array([0.4, 0.4269, -0.0227, -0.0839]) #limb darkening
    theta_star[8:12]= np.array([0.4, 0.4269, -0.0227, -0.0839])


    Nspots = np.random.randint(10)+2
    theta_spot = np.zeros((8, Nspots))
    for k in range(Nspots):
        theta_spot[0,k]=np.random.random()*np.pi #longitude
        theta_spot[1,k]=np.random.random()*np.pi/2 #latitude
        theta_spot[2,k]=np.random.lognormal(mean=np.log(15), sigma=0.5)*np.pi/180 #alpha_max (spot size)
        theta_spot[3,k]=np.random.random()*0.4 + 0.5 #fspot (contrast) 
        theta_spot[4,k]=np.random.random()*(tmax-tmin) + tmin #tmax
        theta_spot[5,k]=np.random.random()*(tmax-tmin) #lifetime
        theta_spot[6,k]=np.random.random()*(tmax-tmin) #ingress
        theta_spot[7,k]=np.random.random()*(tmax-tmin) #egress

    theta_inst = np.array([1., 1.])
    flux = macula(t, theta_star, theta_spot, theta_inst, derivatives=True, temporal=True, tdeltav=True, tstart=t.min()-1, tend=t.max()+10)
    
    #Add noise
    noiselevel = 10**-(np.random.random()*8+4)
    flux += np.random.normal(loc=0, scale=noiselevel, size=flux.size)
    
    #Add flare flags
    is_flare = np.zeros_like(t, dtype=bool)
    
    #Add flares
    #Flare rate will be something between 0- 2/day
    flare_rate = np.random.randint(10,  2 * (t.max() - t.min())  )
    if flare_rate < 0:
        flare_rate=0

    for _ in range(flare_rate):
        tpeak = np.random.uniform( min(t), max(t) )
        fwhm = np.random.uniform( 0.5/24, 1./24 )
        ampl = np.random.lognormal(mean=-7,sigma= 1) #this was -1 for standard
        flux = flux + aflare1( t, tpeak, fwhm, ampl)
        #Change flare flags
#        is_flare[aflare1( t, tpeak, fwhm, ampl) > 1e-3] = True #this was 1e-2
        is_flare[ (t>tpeak-fwhm)*(t<tpeak+2*fwhm)] = True
        is_flare[aflare1( t, tpeak, fwhm, ampl) > 0.0010] = True
 


    # Gaps in data
    Ngaps = np.random.randint(5)
    #Ngaps = 0
    for i in range(Ngaps):
        gap_start = np.random.random() * (t[-1] - t[0])
        gap_length = 0.1 + np.random.normal(loc=2, scale=1)
        flux[np.where((t>gap_start) & (t<gap_start+gap_length))] = 'nan'
 

    #Add random bad points
    #Number of bad points will be ~0.05%
    bad_index = np.random.choice( t.size, size= 
                 np.int( np.random.lognormal( mean=np.log(0.05/100), sigma=0.5 ) * t.size ) )
    fb = np.zeros_like(flux) 
    fb[bad_index] += np.random.normal(loc=0, scale= 10**-(np.random.random()*1+0), size=bad_index.size)
    #Don't consider these flares, even if it's during an event
    is_flare[bad_index] = False
    
    outfile = "%06d.dat" % star
    print("Prot: %0.3f\tNspots: %2d\tnoiselevel: %e\tflares: %d\toutfile: %s" % 
          (periods[star], Nspots, noiselevel, flare_rate, outfile),
		  flush=True)

    np.savetxt("train/"+outfile, np.array([t,flux, is_flare]).T, fmt='%0.5f %0.5f %d' )





