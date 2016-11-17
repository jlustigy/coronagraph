import numpy as np
import scipy as sp
from scipy import special
from numba import jit
import os

__all__ = ["Fstar", "Fplan", "FpFs", "cplan", "czodi", "cezodi", "cspeck", "cdark",
           "cread", "ccic", "f_airy", "f_airy_int", "ctherm", "ctherm_earth",
           "construct_lam", "set_quantum_efficiency", "set_dark_current",
           "set_read_noise", "set_lenslet", "set_throughput", "set_atmos_throughput",
           "get_thermal_ground_intensity", "exptime_element", "lambertPhaseFunction"]

def Fstar(lam, Teff, Rs, d, AU=False):
    '''
    stellar flux function
    --------
    lam - wavelength (um)
    Teff - effective temperature (K)
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

def cplan(q, fpa, T, lam, dlam, Fplan, D):
    '''
    planet photon count rate
    --------
    q - quantum efficiency
    fpa - fraction of planet flux in Airy pattern
    T - system throughput
    lam - wavelength (um)
    dlam - spectral element width (um)
    Fplan - planetary flux (W/m**2/um)
    D - telescope diameter (m)
    cplan - planet photon count rate (s**-1)
    '''
    hc  = 1.986446e-25 # h*c (kg*m**3/s**2)
    return np.pi*q*fpa*T*(lam*1.e-6/hc)*dlam*Fplan*(D/2.)**2.

def czodi(q, X, T, lam, dlam, D, Mzv, SUN=False, CIRC=False):
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
    SUN - set to use WMO solar spectrum
    CIRC - keyword to use a circular aperture

    czodi - zodiacal light photon count rate (s**-1)
    '''
    hc    = 1.986446e-25 # h*c (kg*m**3/s**2)
    F0V   = 3.6e-8     # zero-mag V-band flux (W/m**2/um)
    FsolV = 1.86e+3    # Solar V-band flux at 1 AU
    if SUN:
        print("Error in czodi")
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
    if CIRC:
        # circular aperture size (arcsec**2)
        Omega = np.pi*(X/2.*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture size (arcsec**2)
        Omega = (X*lam*1e-6/D*180.*3600./np.pi)**2.
    return np.pi*q*T*Omega*dlam*(lam*1.e-6/hc)*(D/2)**2.*rat*F0V*10**(-Mzv/2.5)

def cezodi(q, X, T, lam, dlam, D, r, Fstar, Nez, Mezv, SUN=False, CIRC=False):
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
    SUN - set to use WMO solar spectrum
    CIRC - keyword to use a circular aperture

    cezodi - exozodiacal light photon count rate (s**-1)
    '''
    hc    = 1.986446e-25 # h*c (kg*m**3/s**2)
    F0V   = 3.6e-8     # zero-mag V-band flux (W/m**2/um)
    FsolV = 1.86e+3    # Solar V-band flux at 1 AU
    if SUN:
        print("Error in cezodi")
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
    if CIRC:
        # circular aperture size (arcsec**2)
        Omega = np.pi*(X/2.*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture size (arcsec**2)
        Omega = (X*lam*1e-6/D*180.*3600./np.pi)**2.
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

def cdark(De, X, lam, D, theta, DNhpix, IMAGE=False, CIRC=False):
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
    CIRC - keyword to use a circular aperture

    cdark - dark count rate (s**-1)
    '''
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Npix  = Omega/np.pi/theta**2.
    # If not in imaging mode
    if ~IMAGE:
        Npix = 2*DNhpix*Npix
    return De*Npix

def cread(Re, X, lam, D, theta, DNhpix, Dtmax, IMAGE=False, CIRC=False):
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
    CIRC - keyword to use a circular aperture

    cread - read count rate (s**-1)
    '''
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Npix  = Omega/np.pi/theta**2.
    # If not in imaging mode
    if ~IMAGE:
        Npix = 2*DNhpix*Npix
    return Npix/(Dtmax*3600.)*Re

def ccic(Rc, X, lam, D, theta, DNhpix, Dtmax, IMAGE=False, CIRC=False):
    """
    Clock induced charge count rate
    --------
    Rc - clock induced charge counts per pixel per read
    X - diameter or length of photometric aperture (lambda/D)
    lam - wavelength (um)
    D - telescope diameter (m)
    theta - angular diameter of lenslet or pixel (arcsec)
    Dtmax - maximum exposure time (hr)
    IMAGE - keyword set to indicate imaging mode (not IFS)
    CIRC - keyword to use a circular aperture
    cread - read count rate (s**-1)
    """
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Npix  = Omega/np.pi/theta**2.
    # If not in imaging mode
    if ~IMAGE:
        Npix = 2*DNhpix*Npix
    return Npix/(Dtmax*3600.)*Rc


def f_airy(X, CIRC=False):
    """
    fraction of Airy power contained in square or circular aperture
    --------
    X - size of photometric aperture (lambda/D)
    CIRC - keyword to use a circular aperture

    f_airy - fraction of power in Airy pattern of size X*lambda/D
    """
    if CIRC:
        # Circular aperture
        # fraction of power in Airy disk to X*lambda/D
        fpa = 1. - special.jv(0,np.pi*X)**2. - special.jv(1,np.pi*X)**2.
    else:
        # Square aperture
        X_grid   = np.arange(100)/10.
        # fpa on grid, computed using code below
        fpa_grid = \
            [ 0.00000,    0.0309038,     0.117701,     0.244285,     0.388709,     0.528808,\
             0.647304,     0.734911,     0.790813,     0.820765,     0.833866,     0.839249,\
             0.843710,     0.850753,     0.860930,     0.872944,     0.884901,     0.895255,\
             0.903243,     0.908880,     0.912712,     0.915504,     0.917992,     0.920720,\
             0.923959,     0.927698,     0.931686,     0.935540,     0.938892,     0.941519,\
             0.943427,     0.944836,     0.946081,     0.947471,     0.949178,     0.951186,\
             0.953330,     0.955383,     0.957163,     0.958591,     0.959709,     0.960639,\
             0.961530,     0.962501,     0.963607,     0.964837,     0.966128,     0.967391,\
             0.968543,     0.969532,     0.970352,     0.971048,     0.971696,     0.972377,\
             0.973147,     0.974014,     0.974941,     0.975857,     0.976696,     0.977418,\
             0.978026,     0.978559,     0.979075,     0.979627,     0.980241,     0.980912,\
             0.981611,     0.982297,     0.982935,     0.983505,     0.984008,     0.984464,\
             0.984904,     0.985360,     0.985853,     0.986387,     0.986949,     0.987510,\
             0.988040,     0.988519,     0.988944,     0.989331,     0.989707,     0.990099,\
             0.990524,     0.990983,     0.991459,     0.991931,     0.992376,     0.992783,\
             0.993154,     0.993502,     0.993844,     0.994197,     0.994571,     0.994966,\
             0.995376,     0.995784,     0.996176,     0.996542]
        fpa = np.interp(X, X_grid, fpa_grid)
    return fpa

def f_airy_int(X):
    """
    numerical integration to determine fraction of Airy power
    contained square aperture (SLOW!)
    --------

    X - size of photometric aperture (lambda/D)

    f_airy - fraction of Airy power in aperture of size X*lambda/D
    """
    N = 1000      # sets horizontal and vertical grid resolution
    E0 = 1.27324  # total power contained in Airy pattern
    xg = X * np.arange(N) / (N-1)   # horizontal grid
    yg = X * np.arange(N) / (N-1)   # vertical grid
    Iairy = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            # compute Airy intensity
            Iairy[i,j] = 4. * special.jv(1,np.pi*np.sqrt(xg[i]**2. + yg[j]**2.))**2. \
            /(np.pi*np.sqrt(xg[i]**2. + yg[j]**2.))**2.
    Iairy[0,0] = 1.0
    E     = 0.0
    # area of each dxdy element
    dA    = (X/N)**2.
    for i in range(N):
        for j in range(N):
            # sum intensities
            E = E + Iairy[i,j]*dA
    E = 4.*E # factor of 4 as integral only over one quadrant
    fpa   = E/E0

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

def ctherm_earth(q, X, lam, dlam, D, Itherm):
    '''
    telescope thermal count rate
    --------
        q - quantum efficiency
        X - size of photometric aperture (lambda/D)
        lam - wavelength (um)
        dlam - spectral element width (um)
        D - telescope diameter (m)
        Itherm - Earth thermal intensity [W/m**2/um/sr]
    cthe - telescope thermal photon count rate (s**-1)
    '''
    hc    = 1.986446e-25  # h*c (kg*m**3/s**2)
    Omega = np.pi*(X*lam*1.e-6/D)**2. # aperture size (sr**2)
    return np.pi*q*dlam*Itherm*Omega*(lam*1.e-6/hc)*(D/2)**2.

def lambertPhaseFunction(alpha):
    """
    Calculate the Lambertian Phase Function from the phase angle.

    Parameters
    ----------
    alpha: float
        Planet phase angle (degrees)

    Returns
    -------
    Phi : float
        Lambertian phase function
    """
    alpha = alpha * np.pi / 180.
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi

@jit
def construct_lam(lammin, lammax, Res):
    """
    Construct wavelength grid.

    Parameters
    ----------
    lammin : float
        Minimum wavelength [microns]
    lammax : float
        Maximum wavelength [microns]
    Res : float
        Resolving power (lambda / delta-lambda)
    """
    # Set wavelength grid
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
    # Set wavelength widths
    for j in range(1,Nlam-1):
        dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])
    #widths at edges are same as neighbor
    dlam[0] = dlam[1]
    dlam[Nlam-1] = dlam[Nlam-2]
    return lam, dlam

@jit
def set_quantum_efficiency(lam, qe, NIR=False, qe_nir=0.9):
    """

    Parameters
    ----------
    NIR : bool (optional)
        Use near-IR detector proporties
    q_nir : float (optional)
        NIR quantum efficiency
    """
    Nlam = len(lam)
    q = np.zeros(Nlam)
    for j in range(Nlam):
        if (lam[j] <= 0.7):
            q[j] = qe
        else:
            q[j] = qe*(1.0 - (lam[j]-0.7)/(1.0-0.7))
    if q[j] < 0.0: q[j] = 0.

    if NIR:
        iNIR  = (lam > 1.0)
        q[iNIR]  = qe_nir

    return q

def set_dark_current(lam, De, lammax, Tdet, NIR=False, De_nir=1e-3):
    """
    Set grid for dark current as a function of wavelength

    Parameters
    ----------
    lam : ndarray
        Wavelength grid [microns]
    De : float
        Dark current count rate per pixel (s**-1)
    lammax : float
        Maximum wavelength
    Tdet : float
        Detector Temperature [K]
    NIR : bool (optional)
        Use near-IR detector proporties
    De_nir : float (optional)
        NIR minimum dark current count rate per pixel
    """
    De = np.zeros(len(lam)) + De

    if NIR:
        iNIR  = (lam > 1.0)
        # Set dark current based on NIR detector properties
        if ( lammax <= 2.0 ): De[iNIR] = De_nir * np.power(10., (Tdet-120.)*7./100. )
        if ( lammax > 2.0 ) and ( lammax <= 4.0 ): De[iNIR] = De_nir * np.power(10., (Tdet-80.)*9./140. )
        if ( lammax > 4.0 ) and ( lammax <= 7.0 ): De[iNIR] = De_nir * np.power(10., (Tdet-40.)*11./140. )
        if ( lammax > 7.0 ): De[iNIR] = De_nir * np.power(10., (Tdet-30.)*11./70. )
        # Don't let dark current fall below a threshold
        iDe = (De[iNIR] < De_nir)
        De[iNIR][iDe] = De_nir

    return De

def set_read_noise(lam, Re, NIR=False, Re_nir=2.):
    """
    Set grid for read noise as a function of wavelength

    Parameters
    ----------
    lam : ndarray
        Wavelength grid [microns]
    Re : float
        Read noise counts per pixel (s**-1)
    NIR : bool (optional)
        Use near-IR detector proporties
    Re_nir : float (optional)
        NIR read noise counts per pixel
    """
    Re = np.zeros(len(lam)) + Re

    if NIR:
        iNIR  = (lam > 1.0)
        Re[iNIR] = Re_nir

    return Re

def set_lenslet(lam, lammin, diam,
                NIR=False, lammin_nir=1.0):
    """
    Set the angular size of the lenslet

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    lammin : float
        Minimum wavelength
    diam : float
        Telescope Diameter [m]
    NIR : bool (optional)
        Use near-IR detector proporties
    lammin_nir : float (optional)
        Wavelength min to use for NIR lenslet size
    """
    Nlam = len(lam)
    if NIR:
        theta = np.zeros(Nlam)
        iVIS  = (lam <= 1.0)
        iNIR  = (lam > 1.0)
        theta[iVIS] = lammin/1e6/diam/2.*(180/np.pi*3600.)
        # If there are wavelength bins longer than 1um:
        theta[iNIR] = lammin_nir/1e6/diam/2.*(180/np.pi*3600.)
    else:
        theta = lammin/1.e6/diam/2.*(180/np.pi*3600.) # assumes sampled at ~lambda/2D (arcsec)

    return theta

def set_throughput(lam, Tput, diam, sep, IWA, OWA, lammin,
                   FIX_OWA=False, SILENT=False):
    """
    Set wavelength dependent telescope throughput

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    Tput : float
        Throughput
    diam : float
        Telescope diameter [m]
    sep : float
        Planet-star separation in radians
    IWA : float
        Inner working angle
    OWA : float
        Outer working angle
    lammin : float
        Minimum wavelength
    FIX_OWA : bool

    SILENT : bool
        Suppress printing
    """
    Nlam = len(lam)
    T    = Tput + np.zeros(Nlam)
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
    return T

def set_atmos_throughput(lam, dlam, convolve, plot=False):
    """
    Use the atmospheric transmission to set a throughput term for radiation
    through the atmosphere.

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    dlam : ndarray
        Wavelength bin width grid
    convolve : func
        Function used to degrade/downbin spectrum
    """
    # Read in earth transmission file
    fn = os.path.join(os.path.dirname(__file__), "ground/earth_transmission_atacama_30deg.txt")
    tdata = np.genfromtxt(fn, skip_header=5)
    wl_atmos = tdata[:,0]
    Tatmoshr = tdata[:,1]
    # Degrade atmospheric transmission to wavelength gridpoints
    Tatmos = convolve(Tatmoshr, wl_atmos,lam,dlam=dlam)
    if plot:
        import matplotlib.pyplot as plt; from matplotlib import gridspec
        fig1 = plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(1,1)
        ax1 = plt.subplot(gs[0])
        ax1.plot(lam, Tatmos, c="orange", ls="steps-mid")
        ax1.set_ylabel("Earth Atmospheric Transmission")
        ax1.set_xlabel("Wavelength [um]")
        plt.show()
    return Tatmos

def get_thermal_ground_intensity(lam, dlam, convolve):
    """
    Get the intensity at the ground in each spectral element due to the sky background.

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    dlam : ndarray
        Wavelength bin width grid
    convolve : func
        Function used to degrade/downbin spectrum
    """
    # Read in earth thermal data
    fn = os.path.join(os.path.dirname(__file__), "ground/earth_thermal_atacama_30deg.txt")
    tdata = np.genfromtxt(fn, skip_header=6)
    wl_therm = tdata[:,0]  # um
    Fthermhr = tdata[:,1]  # W/m^2/um
    # Degrade earth thermal flux
    Ftherm = convolve(Fthermhr, wl_therm,lam,dlam=dlam)
    # Compute intensity
    Itherm  = Ftherm / np.pi
    return Itherm

def exptime_element(lam, cp, cn, wantsnr):
    """
    Calculate the exposure time (in hours) to get a specified signal-to-noise.

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    cp : ndarray
        Planetary photon count rate [s**-1]
    cn : ndarray
        Noise photon count rate [s**-1]
    wantsnr : float
        Signal-to-noise required in each spectral element

    Returns
    -------
    DtSNR : ndarray
        Exposure time necessary to get specified SNR [hours]
    """
    DtSNR = np.zeros(len(lam))
    i = (cp > 0.)
    j = (cp <= 0.0)
    DtSNR[i] = (wantsnr**2.*cn[i])/cp[i]**2./3600. # (hr)
    DtSNR[j] = np.inf
    return DtSNR
