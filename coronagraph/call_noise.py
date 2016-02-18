import numpy as np
from make_noise import make_noise
from teleplanstar import Star
import pdb

def call_noise(telescope,planet,Ahr='',lamhr=''):
    '''
    (Ahr='', lamhr='', planet='earth', startype='sun', \
    lammin=0.3,lammax=2.0, Res=70., Tput=0.2, diam=8.0, Tsys=274.,\
    IWA=0.5, OWA=30000., d=10., Nez=1., emis=0.9, wantsnr=10.,\
    C=1.e-10, alpha=90., Phi=1./np.pi, planetdir='planets/', \
    Rp=1.0, r=1.0):
    '''
    '''
    planet choices: 
        earth, venus, archean,
        earlymars, hazyarchean, earlyvenus,
        jupiter, saturn, uranus, neptune, mars,
        fstarozone
    star choices (for now):
    sun, f2v
    '''

    whichplanet = planet.name
    startype = planet.star
    planetdir = '../planets/'

    if Ahr == '' and lamhr == '':
        
        if whichplanet == 'earth':
            fn = 'earth_quadrature_radiance_refl.dat'
            model = np.loadtxt(planetdir+fn, skiprows=8)
            lamhr = model[:,0] 
            radhr = model[:,1] 
            solhr = model[:,2] 
            reflhr = model[:,3]
            Ahr   = np.pi*(np.pi*radhr/solhr) # hi-resolution reflectivity
            planet.Rp    = 1.0     # Earth radii
            planet.r     = 1.0     # semi-major axis (AU) 

        if whichplanet == 'venus':
            fn = pldir+'Venus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 0.95     #Earth radii
            planet.r     = 0.72     #semi-major axis (AU)

        if whichplanet == 'archean':
            fn = pldir+'ArcheanEarth_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]     	
            planet.Rp    = 1.0     #Earth radii
            planet.r     = 1.0     #semi-major axis (AU)

        if whichplanet == 'earlymars':
            fn = pldir+'EarlyMars_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 0.53     #Earth radii
            planet.r     = 1.52     #semi-major axis (AU)

        if whichplanet == 'hazyarchean':
            fn = pldir+'Hazy_ArcheanEarth_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]     	
            planet.Rp    = 1.0     #Earth radii
            planet.r     = 1.0     #semi-major axis (AU)

        if whichplanet == 'earlyvenus':
            fn = pldir+'EarlyVenus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 0.95     #Earth radii
            planet.r     = 0.72     #semi-major axis (AU)

        if whichplanet == 'jupiter':
            fn = pldir+'Jupiter_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 10.86     #Earth radii
            planet.r     = 5.20     #semi-major axis (AU)

        if whichplanet == 'saturn':
            fn = pldir+'Saturn_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 9.00     #Earth radii
            planet.r     = 9.54     #semi-major axis (AU)

        if whichplanet == 'uranus':
            fn = pldir+'Uranus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 3.97     #Earth radii
            planet.r     = 19.19     #semi-major axis (AU)


        if whichplanet == 'warmuranus':
            fn = pldir+'Uranus_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 3.97     #Earth radii
            planet.r     = 5.20     #semi-major axis (AU)
        
        if whichplanet == 'warmneptune':
            fn = pldir+'Neptune_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 3.97     #Earth radii
            planet.r     = 5.20     #semi-major axis (AU)

        if whichplanet == 'neptune':
            fn = pldir+'Neptune_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 3.85     #Earth radii
            planet.r     = 30.07     #semi-major axis (AU)


        if whichplanet == 'mars':
            fn = pldir+'Mars_geo_albedo.txt'
            model = np.loadtxt(planetdir+fn) 
            lamhr = model[:,0] 
            Ahr   = model[:,1]
            planet.Rp    = 0.53     #Earth radii
            planet.r     = 1.52     #semi-major axis (AU)

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
    
    star = Star(Teff=Teff, Rs=Rs)
 
    # Shawn: "I don't like noise.  It makes me sad."

    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR \
        = make_noise(Ahr, lamhr, telescope, planet, star, COMPUTE_LAM=True)

    return lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR