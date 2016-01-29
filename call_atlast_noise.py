'''
call_atlast_noise.py

Author
--------
This code is a python reproduction of the IDL code 
run_atlast_noise.pro by Ty Robinson. 

Python code by Jacob Lustig-Yaeger

Parameters
--------
planet : string
    Name of planet to observe with telescope model
    Ex: 'earth', 'venus', 'warmuranus'
startype : string
    Type of star (lowercase)
lammin : float
    Telescope minimum wavlength (um)
lammax : float
    Telescope maximum wavlength (um)
res : float
    Telescope spectral resolution (lambda/Dlambda)
Tput : float
    Telescope throughput
diam : float
    Telescope diameter (m)
Tsys : float
    Telescope temperature (K)
IWA : float
    Telescope inner working angle (lambda/D)
OWA : float
    Telescope outer working angle (lambda/D)
d : float
    Distance to planet (pc)
Nez : float
    Number of exo-zodis (SS zodi)
emis : float
    Telescope/system emissivity
wantsnr : float
    Desired Signal-to-Noise of observation
C : float
    Raw Contrast
alpha : float
    Phase angle (default 90, quadrature)
Phi : float
    Phase function (default 1/pi, quadrature)
planetdir : string
    Directory housing precomputed planet albedo spectra
Rp : float
    Planet radius (default 1.0, Earth)
r : float
    Planet semi-major axis (default 1.0, Earth)

Note
--------
All parameters are set to the current design specifications
for the ATLAST mission. They can be adjusted as needed for 
modeling variants on this concept. 
'''

import numpy as np
from atlast_noise import atlast_noise
import pdb

def call_atlast_noise(Ahr='', lamhr='', planet='earth', startype='sun', \
    lammin=0.3,lammax=2.0, Res=70., Tput=0.2, diam=8.0, Tsys=274.,\
    IWA=0.5, OWA=30000., d=10., Nez=1., emis=0.9, wantsnr=10.,\
    C=1.e-10, alpha=90., Phi=1./np.pi, planetdir='planets/', \
    Rp=1.0, r=1.0):

    '''
    planet choices: 
        earth, venus, archean,
        earlymars, hazyarchean, earlyvenus,
        jupiter, saturn, uranus, neptune, mars,
        fstarozone
    star choices (for now):
    sun, f2v
    '''

    whichplanet = planet

    if Ahr == '' and lamhr == '':
        
        if whichplanet == 'earth':
            fn = 'earth_quadrature_radiance_refl.dat'
            model = np.loadtxt(planetdir+fn, skiprows=8)
            lamhr = model[:,0] 
            radhr = model[:,1] 
            solhr = model[:,2] 
            reflhr = model[:,3]
            Ahr   = np.pi*(np.pi*radhr/solhr) # hi-resolution reflectivity
            Rp    = 1.0     # Earth radii
            r     = 1.0     # semi-major axis (AU) 

        if whichplanet == 'venus':
            fn = 'Venus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 0.95     #Earth radii
            r     = 0.72     #semi-major axis (AU)

        if whichplanet == 'archean':
            fn = 'ArcheanEarth_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]     	
            Rp    = 1.0     #Earth radii
            r     = 1.0     #semi-major axis (AU)

        if whichplanet == 'earlymars':
            fn = 'EarlyMars_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 0.53     #Earth radii
            r     = 1.52     #semi-major axis (AU)

        if whichplanet == 'hazyarchean':
            fn = 'Hazy_ArcheanEarth_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]     	
            Rp    = 1.0     #Earth radii
            r     = 1.0     #semi-major axis (AU)

        if whichplanet == 'earlyvenus':
            fn = 'EarlyVenus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 0.95     #Earth radii
            r     = 0.72     #semi-major axis (AU)

        if whichplanet == 'jupiter':
            fn = 'Jupiter_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 10.86     #Earth radii
            r     = 5.20     #semi-major axis (AU)

        if whichplanet == 'saturn':
            fn = 'Saturn_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 9.00     #Earth radii
            r     = 9.54     #semi-major axis (AU)

        if whichplanet == 'uranus':
            fn = 'Uranus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 3.97     #Earth radii
            r     = 19.19     #semi-major axis (AU)


        if whichplanet == 'warmuranus':
            fn = 'Uranus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 3.97     #Earth radii
            r     = 5.20     #semi-major axis (AU)
        
        if whichplanet == 'warmneptune':
            fn = 'Neptune_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 3.97     #Earth radii
            r     = 5.20     #semi-major axis (AU)

        if whichplanet == 'neptune':
            fn = 'Neptune_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 3.85     #Earth radii
            r     = 30.07     #semi-major axis (AU)


        if whichplanet == 'mars':
            fn = 'Mars_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            Rp    = 0.53     #Earth radii
            r     = 1.52     #semi-major axis (AU)

        if whichplanet == 'fstarozone':
            print('fstarozone functionality not yet added') 
            #fn = 'fstarcloudy.sav'
            #fn = 'F2V_5.e-1fCO2_1.e6H2Volc_1.e10BIF.out_toa.rad'
            #READCOL, 'planets/'+fn, lamhr, wnhr, sunhr, fluxhr
            #restore, 'planets/'+fn
            #lamhr = reverse(wl_)
            #ahr = reverse(cloudyalb_)
            #Ahr = (2/3.) * fluxhr/(sunhr/2.) #geometric albedo & phase corrections
            #Rp    = 1.0     #Earth radii
            #r     = 1.72    #semi-major axis (AU)



    # star parameters
    if startype == '':
        Teff  = 5780.   #Sun-like Teff (K)
        Rs    = 1.      #star radius in solar radii
    if  startype == 'sun':
        Teff  = 5780.   #Sun-like Teff (K)
        Rs    = 1.      #star radius in solar radii
    if  startype == 'f2v':
        Teff  = 7050.   #Sun-like Teff (K)
        Rs    = 1.3      #star radius in solar radii
 
    # Shawn: "I don't like noise.  It makes me sad."

    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR \
        = atlast_noise(Ahr, lamhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
        lammin, lammax, Res, diam, Tput, C, IWA, OWA, Tsys, emis,\
        wantsnr, whichplanet, COMPUTE_LAM=True)
    #pdb.set_trace()
    return lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR



