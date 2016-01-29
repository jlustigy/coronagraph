'''
atlast_noise.py

Author
--------
This code is a python reproduction of the IDL code 
atlast_noise.pro by Ty Robinson. 

Python code by Jacob Lustig-Yaeger

Parameters:
--------
Ahr  - hi-res planetary albedo spectrum
lamhr  - wavelength grid for Ahr (um)
alpha  - phase angle (deg)
Phi  - phase function evaluated at alpha
Rp  - planetary radius (Rearth)
Teff  - stellar effective temperature (K)
Rs  - stellar radius (Rsun)
r  - orbital separation (au)
d  - distance to system (pc)
Nez  - number of exozodis
lammin - minimum wavelength (um)
lammax - maximum wavelength (um)
Res - spectral resolution (lambda/Dlambda)
diam - telescope diameter (m)
Tput - system throughput
C  - raw contrast
IWA - inner working angle (lambda/D)
OWA - outer working angle (lambda/D; unless /FIXOWA)
Tsys - observing system temperature for thermal background (K)
emis - observing system emissivity

Returns:
--------
lam  - low-res wavelength grid (um)
dlam  - spectral bin widths (um)
A  - planetary albedo spectrum at low-res
q  - quantum efficiency
Cratio  - planet-star contrast (flux) ratio
cp  - planet count rate (s**-1)
csp  - speckle count rate (s**-1)
cz  - zodiacal light count rate (s**-1) 
cez  - exozodiacal light count rate (s**-1)
cD  - dark current count rate (s**-1)
cR  - read noise count rate (s**-1)
cth  - internal thermal count rate (s**-1)
DtSNR  - integration time for SNR (hr)
wantsnr - SNR you want the integration time for

Options:
--------
FIX_OWA - set to fix OWA at OWA*lammin/D, as would occur 
    if lenslet array is limiting the OWA
COMPUTE_LAM - set to compute lo-res wavelength grid, otherwise 
      the grid input as variable 'lam' is used
SILENT - set to silence all warnings
'''

# Import dependent modules
import numpy as np
from degrade_spec import degrade_spec
from noise_routines import *
import pdb

def atlast_noise(Ahr, lamhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
    lammin, lammax, Res, diam, Tput, C, IWA, OWA, Tsys, emis,\
    wantsnr, whichplanet, FIX_OWA = False, COMPUTE_LAM = False,\
    SILENT = False):

    # Set key system parameters
    De     = 1.e-4  # dark current (s**-1)
    DNHpix = 3      # horizontal pixel spread of IFS spectrum
    Re     = 0.1    # read noise per pixel
    Dtmax  = 1.     # maximum exposure time (hr)
    X      = 0.7    # size of photometric aperture (lambda/D)
    q      = 0.9    # quantum efficiency

    # Set astrophysical parameters
    MzV  = 23.0     # zodiacal light surface brightness (mag/arcsec**2)
    MezV = 22.0     # exozodiacal light surface brightness (mag/arcsec**2)

    # Compute angular size of lenslet
    theta = lammin/1.e6/diam/2.*(180/np.pi*3600.) #assumes sampled at ~lambda/2D (arcsec)

    # Set wavelength grid
    if COMPUTE_LAM:
        lam  = lammin #in [um]
        Nlam = 1
        while (lam < lammax):
            lam  = lam + lam/Res
            Nlam = Nlam +1 
        lam    = np.zeros(Nlam)
        lam[0] = lammin
        for j in range(1,Nlam):
            lam[j] = lam[j-1] + lam[j-1]/Res
    Nlam = len(lam)
    dlam = np.zeros(Nlam) #grid widths (um)
    for j in range(1,Nlam-1):
        dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])
    #widths at edges are same as neighbor
    dlam[0] = dlam[1]
    dlam[Nlam-1] = dlam[Nlam-2]

    # Set throughput      
    T    = np.zeros(Nlam)
    T[:] = Tput
    sep  = r/d*np.sin(alpha*np.pi/180.)*np.pi/180./3600. # separation in radians
    iIWA = ( sep < IWA*lam/diam/1.e6 )
    if (True if True in iIWA else False):
          T[iIWA] = 0. #zero transmission for points inside IWA have no throughput
          if ~SILENT: 
              print 'WARNING: portions of spectrum inside IWA'
    if FIX_OWA:
          if ( sep > OWA*lammin/diam/1.e6 ):
            T[:] = 0. #planet outside OWA, where there is no throughput
            if ~SILENT: 
                print 'WARNING: planet outside fixed OWA'
    else:
          iOWA = ( sep > OWA*lam/diam/1.e6 )
          if (True if True in iOWA else False):
            T[iOWA] = 0. #points outside OWA have no throughput
            if ~SILENT:
                print 'WARNING: portions of spectrum outside OWA'


    # Degrade albedo spectrum
    if COMPUTE_LAM:
        A = degrade_spec(Ahr,lamhr,lam,dlam=dlam)
    else: 
        A = Ahr

    # Compute fluxes
    Fs = Fstar(lam, Teff, Rs, r, AU=True) # stellar flux on planet
    Fp = Fplan(A, Phi, Fs, Rp, d)         # planet flux at telescope
    Cratio = FpFs(A, Phi, Rp, r)

      
    # Compute count rates
    cp     =  cplan(q, X, T, lam, dlam, Fp, diam)     # planet count rate
    cz     =  czodi(q, X, T, lam, dlam, diam, MzV)    # solar system zodi count rate
    cez    =  cezodi(q, X, T, lam, dlam, diam, r, \
        Fstar(lam,Teff,Rs,1.,AU=True), Nez, MezV)         # exo-zodi count rate
    csp    =  cspeck(q, T, C, lam, dlam, Fstar(lam,Teff,Rs,d), diam) # speckle count rate
    cD     =  cdark(De, X, lam, diam, theta, DNHpix)  # dark current count rate
    cR     =  cread(Re, X, lam, diam, theta, DNHpix, Dtmax)  # readnoise count rate
    cth    =  ctherm(q, X, lam, dlam, diam, Tsys, emis) # internal thermal count rate
    cnoise =  cp + 2*(cz + cez + csp + cD + cR + cth)   # assumes background subtraction
    cb = (cz + cez + csp + cD + cR + cth)
    ctot = cp + cz + cez + csp + cD + cR + cth
    '''
    Giada: where does the factor of 2 come from?

    Ty (Via email): That's due to "background subtraction".  
    If you were to take a single exposure, and had the ability 
    to post-process the image to the Poisson noise limit, you 
    wouldn't have the factor of two.  However, it's not yet 
    clear that we'll be able to reach the Poisson, single observation limit.  
    Instead, the current idea is that you take two observations (with 
    exposure time Delta t/2), with the telescope rotated by a 
    small amount between exposures, and then subtract the two images.  
    So, for a fixed exoplanet count (cp), the roll technique would 
    give you 2x as many noise counts due to background sources as 
    would the single-observation technique. 
    See also the top of page 4 of Brown (2005).
    '''    

    # Exposure time to SNR    
    DtSNR = np.zeros(Nlam)
    DtSNR[:] = 0.
    i = (cp > 0.)
    if (True if True in i else False): 
        DtSNR[i] = (wantsnr**2.*cnoise[i])/cp[i]**2./3600. # (hr)
    # added by Giada:
    #if whichplanet == 'earth':
    #    print 'Functionality not added to python version... yet.'
        #pt5 = closest(lam, 0.55) ;same wavelength chris stark used
        #time = dtsnr(pt5)*3600.*1
        #save, time, filename='~/idl/noise_model/earthtime.sav'
    #if whichplanet != 'earth': 
    #    print 'Functionality not added to python version... yet.'    
        #then restore, '~/idl/noise_model/earthtime.sav'
         
    # These pieces are fundamental, but should go outside this function
    # as they depend on the particular exposure time, not just the telescope    
    #noisyspec = np.random.poisson(cnoise * time)
    #planet = noisyspec - 2.*(cz + cez + csp + cD + cR + cth)*  time
    #sun = (time * cp)/A
    #SNR = cp*time/np.sqrt((cp+2*cb)*time)
    #noisy = np.random.randn(len(A))*A/SNR+A

    return lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR











