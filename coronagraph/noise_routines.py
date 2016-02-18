import numpy as np
import scipy as sp
from scipy import special

def Fstar(lam, Teff, Rs, d, AU=False):
    '''
    stellar flux function
    --------
      lam - wavelength (um)
       Teff - effective tem
       perature (K)
         Rs - stellar radius (solar radii)
      d - distance to star (pc)
         AU - flag that indicates d is in AU
    Fstar - stellar flux (W/m**2/um)
    '''
    Rsun  = 6.958e8        # solar radius (m)
    ds    = 3.08567e16     # parsec (m)
    if AU:
        ds = 1.495979e11     # AU (m)
    lam= 1.e-6 * lam        # wavelength (m)
    c1    = 3.7417715e-16    # 2*pi*h*c*c (kg m**4 / s**3)   
    c2    = 1.4387769e-2     # h*c/k (m K)
    power   = c2/lam/Teff     # (unitless) 
    Fs    = c1/( (lam**5.)*(np.exp(power)-1.) ) * 1.e-6
    return Fs*(Rs*Rsun/d/ds)**2.

def Fplan(A, Phi, Fstar, Rp, d, AU=False):
    '''
    planetary flux functiom
    --------
      A - planetary geometric albedo
        Phi - planetary phase function
      Fstar - stellar flux (W/m**2/um)
         Rp - planetary radius (Earth radii)
      d - distance (pc)
         au - flag that indicates d is in AU
    Fplan - planetary flux (W/**2/um)
    '''
    Re    = 6.371e6        # radius of Earth (m)
    ds    = 3.08567e16     # parsec (m)
    if AU:  
        ds = 1.495979e11     # AU (m)    
    return A*Phi*Fstar*(Rp*Re/d/ds)**2.

def FpFs(A, Phi, Rp, r):
    '''
    planet-star flux ratio
    --------
      A - planetary geometric albedo
      Phi - planetary phase function
      Rp - planetary radius (Earth radii)
      r - orbital distance (au)
     FpFs - planet-star flux ratio
     '''
    Re = 6.371e6         # radius of Earth (m)
    ds = 1.495979e11       # AU (m)
    return A*Phi*(Rp*Re/r/ds)**2.    

def cplan(q, X, T, lam, dlam, Fplan, D):
    '''
    planet photon count rate
    --------
    q - quantum efficiency
    X - size of photometric aperture (lambda/D)
    T - system throughput
    lam - wavelength (um)
    dlam - spectral element width (um)
    Fplan - planetary flux (W/m**2/um)
    D - telescope diameter (m)
    cplan - planet photon count rate (s**-1)
    '''
    hc  = 1.986446e-25 # h*c (kg*m**3/s**2)
    fpa = 1. - special.jv(0,np.pi*X)**2. - special.jv(1,np.pi*X)**2. # fraction of power in Airy disk to X*lambda/D
    return np.pi*q*fpa*T*(lam*1.e-6/hc)*dlam*Fplan*(D/2)**2.

def czodi(q, X, T, lam, dlam, D, Mzv, SUN=False):
    '''
    zodiacal light count rate
    --------
      q - quantum efficiency
        X - size of photometric aperture (lambda/D)
        T - system throughput
      lam - wavelength (um)
         dlam - spectral element width (um)
        D - telescope diameter (m)
      MzV - zodiacal light surface brightness (mag/arcsec**2)
         /SUN - set to use WMO solar spectrum
    czodi - zodiacal light photon count rate (s**-1)
    '''
    hc    = 1.986446e-25 # h*c (kg*m**3/s**2)
    F0V   = 3.6e-8     # zero-mag V-band flux (W/m**2/um)
    FsolV = 1.86e+3    # Solar V-band flux at 1 AU
    if SUN:
      fn    = '/Users/robinson/Work/noise/wmo_solar_spectrum.dat'
      # Read-in solar spectrum and interpolate it onto lam using degrade_spec()
      # READCOL, fn, lamsol, Fsol, SKIPLINE=32, /SILENT
      # Fsol  = DEGRADE_SPEC(Fsol,lamsol,lam,DLAM=dlam) ; degrade solar spectrum (W/m**2/um)
    else: 
        Teffs  = 5778. # Sun effective temperature
        Rs  = 1.       # Sun radius (in solar radii)
        Fsol  = Fstar(lam, Teffs, Rs, 1., AU=True)
    rat   = np.zeros(len(lam))
    rat[:]= Fsol[:]/FsolV # ratio of solar flux to V-band solar flux
    Omega = np.pi*(X*lam*1.e-6/D*180.*3600./np.pi)**2. # aperture size (arcsec**2)
    return np.pi*q*T*Omega*dlam*(lam*1.e-6/hc)*(D/2)**2.*rat*F0V*10**(-Mzv/2.5)

def cezodi(q, X, T, lam, dlam, D, r, Fstar, Nez, Mezv, SUN=False):
    '''
    exozodiacal light count rate
    --------
      q - quantum efficiency
      X - size of photometric aperture (lambda/D)
      T - system throughput
      lam - wavelength (um)
         dlam - spectral element width (um)
      D - telescope diameter (m)
      r - orbital distance (au)
        Fstar - host star spectrum *at 1 au* (W/m**2/um)
      Nez - number of exozodis
         MezV - exozodiacal light surface brightness (mag/arcsec**2)
         /SUN - set to use WMO solar spectrum
     cezodi - exozodiacal light photon count rate (s**-1)
     '''
    hc    = 1.986446e-25 # h*c (kg*m**3/s**2)
    F0V   = 3.6e-8     # zero-mag V-band flux (W/m**2/um)
    FsolV = 1.86e+3    # Solar V-band flux at 1 AU
    if SUN:
        fn    = '/Users/robinson/Work/noise/wmo_solar_spectrum.dat'
        # Read-in solar spectrum and interpolate it onto lam using degrade_spec()
        # READCOL, fn, lamsol, Fsol, SKIPLINE=32, /SILENT
        # Fsol  = DEGRADE_SPEC(Fsol,lamsol,lam,DLAM=dlam) ; degrade solar spectrum (W/m**2/um)
    else:
        Teffs  = 5778.   # Sun effective temperature
        Rs  = 1.       # Sun radius (in solar radii)
        #Fsol  = Fstar(lam, Teffs, Rs, 1., AU=True)  # Sun as blackbody (W/m**2/um)
    rat   = np.zeros(len(lam))
    rat[:]= Fstar[:]/FsolV # ratio of solar flux to V-band solar flux
    Omega = np.pi*(X*lam*1.e-6/D*180.*3600./np.pi)**2. # aperture size (arcsec**2)
    return np.pi*q*T*Omega*dlam*(lam*1.e-6/hc)*(D/2)**2.*(1./r)**2.*rat*Nez*F0V*10**(-Mezv/2.5)

def cspeck(q, T, C, lam, dlam, Fstar, D):
    '''
    speckle count rate
    --------
      q - quantum efficiency
        T - system throughput
        C - design contrast
      lam - wavelength (um)
         dlam - spectral element width (um)
      D - telescope diameter (m)
        Fstar - host star spectrum at distance to system (W/m**2/um)
     cspeck - speckle photon count rate (s**-1)
     '''
    hc    = 1.986446e-25 # h*c (kg*m**3./s**2.)
    return np.pi*q*T*C*dlam*Fstar*(lam*1.e-6/hc)*(D/2.)**2.

def cdark(De, X, lam, D, theta, DNhpix, IMAGE=False):
    '''
    dark count rate
    --------
       De - dark count rate (s**-1)
        X - size of photometric aperture (lambda/D)
      lam - wavelength (um)
        D - telescope diameter (m)
        theta - angular size of lenslet or pixel (arcsec**2)
       DNhpix - number of pixels spectrum spread over in horizontal, for IFS
        IMAGE - keyword set to indicate imaging mode (not IFS)
        cdark - dark count rate (s**-1)
    '''
    Omega = np.pi*(X*lam*1.e-6/D*180.*3600./np.pi)**2. # aperture size (arcsec**2)
    Npix  = Omega/np.pi/theta**2.
    if ~IMAGE: 
        Npix = 2*DNhpix*Npix
    return De*Npix

def cread(Re, X, lam, D, theta, DNhpix, Dtmax, IMAGE=False):
    '''
    read noise count rate
    --------
       Re - read noise counts per pixel
        X - size of photometric aperture (lambda/D)
      lam - wavelength (um)
        D - telescope diameter (m)
        theta - angular size of lenslet or pixel (arcsec**2)
        Dtmax - maximum exposure time (hr)
        IMAGE - keyword set to indicate imaging mode (not IFS)
    cread - read count rate (s**-1)
    '''
    Omega = np.pi*(X*lam*1.e-6/D*180.*3600./np.pi)**2. # aperture size (arcsec**2)
    Npix  = Omega/np.pi/theta**2.
    if ~IMAGE: 
        Npix = 2*DNhpix*Npix  
    return Npix/(Dtmax*3600.)*Re

def ctherm(q, X, lam, dlam, D, Tsys, emis):
    '''
    telescope thermal count rate
    --------
        q - quantum efficiency
        X - size of photometric aperture (lambda/D)
      lam - wavelength (um)
         dlam - spectral element width (um)
        D - telescope diameter (m)
         Tsys - telescope/system temperature (K)
         emis - telescope/system emissivity
     ctherm - telescope thermal photon count rate (s**-1)
    '''
    hc    = 1.986446e-25  # h*c (kg*m**3/s**2)
    c1    = 3.7417715e-16 # 2*pi*h*c*c (kg m**4 / s**3)   
    c2    = 1.4387769e-2  # h*c/k (m K)
    lambd= 1.e-6*lam     # wavelength (m)
    power   = c2/lambd/Tsys
    Bsys  = c1/( (lambd**5.)*(np.exp(power)-1.) )*1.e-6/np.pi # system Planck function (W/m**2/um/sr)
    Omega = np.pi*(X*lam*1.e-6/D)**2. # aperture size (sr**2)
    return np.pi*q*dlam*emis*Bsys*Omega*(lam*1.e-6/hc)*(D/2)**2.