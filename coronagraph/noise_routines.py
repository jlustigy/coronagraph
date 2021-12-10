'''
Routines for simulating `coronagraph` model noise terms are listed below. These
functions are used within the :class:`coronagraph.CoronagraphNoise` and
:func:`coronagraph.count_rates` functions, but are provided here for
independent use.

The most important functions are the individual photon count rate terms due to
different photon sources. This including photons from the planet :func:`cplan`,
from zodiacal light :func:`czodi` and exo-zodiacal light :func:`cezodi`, from
coronagraph speckles :func:`cspeck`, from dark current :func:`cdark` and read
noise :func:`cread`, from thermal emission from the telescope mirror
:func:`ctherm`, and from clock-induced charge :func:`ccic`.

Optional ground-based telescope noise modeling includes extra terms for the
emission from Earth's atmosphere incident on the telescope,
:func:`ctherm_earth` (also see :func:`get_sky_flux`), and an additional
throughput term due to atmospheric extinction :func:`set_atmos_throughput`.

Finally, there are some extra convenience functions:
Calculate the fraction of Airy power contained in square or circular aperture
using :func:`f_airy`;
Construct a wavelength grid by specifying either a spectral resolving power or
a fixed wavelength bandwidth using :func:`construct_lam`;
Calculate the Lambertian Phase Function of a planet from the phase angle using
:func:`lambertPhaseFunction`;
Calculate the Planck blackbody radiance given temperature and wavelength
using :func:`planck`.
'''

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

import numpy as np
import scipy as sp
import pandas as pd
from scipy import special
from scipy.interpolate import interp1d
import os
import astropy.units as u
from astropy.io import fits

__all__ = ["Fstar", "Fplan", "FpFs", "cplan", "czodi", "cezodi", "cspeck", "cdark",
           "cread", "ccic", "f_airy", "ctherm", "ctherm_earth", "cstar",
           "construct_lam", "set_quantum_efficiency", "set_dark_current",
           "set_read_noise", "set_lenslet", "set_throughput", "set_atmos_throughput",
           "exptime_element", "lambertPhaseFunction",
           "get_sky_flux", "planck", "set_atmos_throughput_skyflux", "set_AO_fpa",
           "calculate_Npix"]

def Fstar(lam, Teff, Rs, d, AU=False):
    """
    Stellar flux function

    Parameters
    ----------
    lam : float or array-like
        Wavelength [um]
    Teff : float
        Stellar effective temperature [K]
    Rs :
        Stellar radius [solar radii]
    d :
        Distance to star [pc]
    AU : bool, optional
        Flag that indicates d is in AU

    Returns
    -------
    Fstar : float or array-like
        Stellar flux [W/m**2/um]
    """
    Rsun  = 6.958e8          # solar radius (m)
    ds    = 3.08567e16       # parsec (m)
    if AU:
        ds = 1.495979e11     # AU (m)
    lam= 1.e-6 * lam         # wavelength (m)
    c1    = 3.7417715e-16    # 2*pi*h*c*c (kg m**4 / s**3)
    c2    = 1.4387769e-2     # h*c/k (m K)
    power   = c2/lam/Teff    # (unitless)
    Fs    = c1/( (lam**5.)*(np.exp(power)-1.) ) * 1.e-6
    return Fs*(Rs*Rsun/d/ds)**2.

def Fplan(A, Phi, Fstar, Rp, d, AU=False):
    """
    Planetary flux function

    Parameters
    ----------
    A : float or array-like
        Planetary geometric albedo
    Phi : float
        Planetary phase function
    Fstar : float or array-like
        Stellar flux [W/m**2/um]
    Rp : float
        Planetary radius [Earth radii]
    d : float
        Distance to star [pc]
    AU : bool, optional
        Flag that indicates d is in AU

    Returns
    -------
    Fplan : float or array-like
        Planetary flux [W/m**2/um]
    """
    Re    = 6.371e6        # radius of Earth (m)
    ds    = 3.08567e16     # parsec (m)
    if AU:
        ds = 1.495979e11     # AU (m)
    return A*Phi*Fstar*(Rp*Re/d/ds)**2.

def FpFs(A, Phi, Rp, r):
    """
    Planet-star flux ratio (Equation 11 from Robinson et al. 2016).

    :math:`\\frac{{F}_{p,\\lambda }}{{F}_{{\\rm{s}},\\lambda }}=A{\\rm{\\Phi }}(\\alpha ){\\left(\\displaystyle \\frac{{R}_{{\\rm{p}}}}{r}\\right)}^{2}`

    Parameters
    ----------
    A : float or array-like
        Planetary geometric albedo
    Phi : float
        Planetary phase function
    Rp : float
        Planetary radius [Earth radii]
    r : float
        Planetary orbital semi-major axis [AU]

    Returns
    -------
    FpFs : float or array-like
        Planet-star flux ratio
    """
    Re = 6.371e6         # radius of Earth (m)
    ds = 1.495979e11       # AU (m)
    return A*Phi*(Rp*Re/r/ds)**2.

def cstar(q, fpa, T, lam, dlam, Fstar, D):
    """
    Stellar photon count rate (not used with coronagraph)

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    fpa : float
        Fraction of planetary light that falls within photometric aperture
    T : float
        Telescope and system throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    Fstar : float or array-like
        Stellar flux [W/m**2/um]
    D : float
        Telescope diameter [m]

    Returns
    -------
    cs : float or array-like
        Stellar photon count rate [1/s]
    """
    hc  = 1.986446e-25 # h*c (kg*m**3/s**2)
    return np.pi*q*fpa*T*(lam*1.e-6/hc)*dlam*Fstar*(D/2.)**2.

def cplan(q, fpa, T, lam, dlam, Fplan, D):
    """
    Exoplanetary photon count rate (Equation 12 from Robinson et al. 2016)

    :math:`{c}_{{\\rm{p}}}=\\pi {{qf}}_{\\mathrm{pa}}{ \\mathcal T }\\displaystyle \\frac{\\lambda }{{hc}}{F}_{{\\rm{p}},\\lambda }(d)\\Delta \\lambda {\\left(\\displaystyle \\frac{D}{2}\\right)}^{2}`

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    fpa : float
        Fraction of planetary light that falls within photometric aperture
    T : float
        Telescope and system throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    Fplan : float or array-like
        Planetary flux [W/m**2/um]
    D : float
        Telescope diameter [m]

    Returns
    -------
    cplan : float or array-like
        Exoplanetary photon count rate [1/s]
    """
    hc  = 1.986446e-25 # h*c (kg*m**3/s**2)
    return np.pi*q*fpa*T*(lam*1.e-6/hc)*dlam*Fplan*(D/2.)**2.

def czodi(q, X, T, lam, dlam, D, Mzv, SUN=False, CIRC=False):
    """
    Zodiacal light count rate (Equation 15 from Robinson et al. 2016)

    :math:`{c}_{{\\rm{z}}}=\\pi q{ \\mathcal T }{\\rm{\\Omega }}\\Delta \\lambda \\displaystyle \\frac{\\lambda }{{hc}}{\\left(\\displaystyle \\frac{D}{2}\\right)}^{2}\\displaystyle \\frac{{F}_{\\odot ,\\lambda }(1\\;{\\rm{AU}})}{{F}_{\\odot ,V}(1\\;{\\rm{AU}})}\\;{F}_{0,V}{10}^{-{M}_{{\\rm{z}},V}/2.5}`

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    X : float
        Size of photometric aperture (lambda/D)
    T : float
        Telescope and system throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    MzV : float
        Zodiacal light surface brightness [mag/arcsec**2]
    SUN : bool, optional
        Set to use solar spectrum (Not Implemented)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    czodi : float or array-like
        Zodiacal light photon count rate [1/s]
    """
    hc    = 1.986446e-25 # h*c (kg*m**3/s**2)
    F0V   = 3.6e-8     # zero-mag V-band flux (W/m**2/um)
    FsolV = 1.86e+3    # Solar V-band flux at 1 AU
    if SUN:
        print("Error in function czodi()")
        raise NotImplementedError
        # Read-in solar spectrum and interpolate it onto lam using degrade_spec()
    else:
        Teffs  = 5778. # Sun effective temperature
        Rs  = 1.       # Sun radius (in solar radii)
        Fsol  = Fstar(lam, Teffs, Rs, 1., AU=True)
    rat   = np.zeros(len(lam))
    rat[:]= Fsol[:]/FsolV # ratio of solar flux to V-band solar flux
    if CIRC:
        # circular aperture size (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture size (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    return np.pi*q*T*Omega*dlam*(lam*1.e-6/hc)*(D/2.)**2.*rat*F0V*10**(-Mzv/2.5)

def cezodi(q, X, T, lam, dlam, D, r, Fstar, Nez, Mezv, SUN=False, CIRC=False):
    """
    Exozodiacal light count rate (Equation 18 from Robinson et al. 2016)

    :math:`{c}_{\\mathrm{ez}} = \\pi q{ \\mathcal T }{X}^{2}\\displaystyle \\frac{{\\lambda }^{4}}{4{hc}{ \\mathcal R }}{\\left(\\displaystyle \\frac{1{\\rm{AU}}}{r}\\right)}^{2}\\displaystyle \\frac{{F}_{{\\rm{s}},\\lambda }(1\\;{\\rm{AU}})}{{F}_{{\\rm{s}},V}(1\\;{\\rm{AU}})}\\\\ \\times \\displaystyle \\frac{{F}_{{\\rm{s}},V}(1\\;{\\rm{AU}})}{{F}_{\\odot ,V}(1\\;{\\rm{AU}})}{N}_{\\mathrm{ez}}{F}_{0,V}{10}^{-{M}_{\\mathrm{ez},V}/2.5}`

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    X : float
        Size of photometric aperture (lambda/D)
    T : float
        System throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    r : float
        Planetary orbital semi-major axis [AU]
    Fstar : array-like
        Host star spectrum *at 1 au* (W/m**2/um)
    Nez : float
        Number of exozodis in exoplanetary disk
    MezV : float
        Exozodiacal light surface brightness [mag/arcsec**2]
    SUN : bool, optional
        Set to use solar spectrum (Not Implemented)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    cezodi : float or array-like
        Exozodiacal light photon count rate [1/s]
    """
    hc    = 1.986446e-25   # h*c (kg*m**3/s**2)
    F0V   = 3.6e-8         # zero-mag V-band flux (W/m**2/um)
    FsolV = 1.86e+3        # Solar V-band flux at 1 AU
    if SUN:
        print("Error in function cezodi()")
        raise NotImplementedError
        # Read-in solar spectrum and interpolate it onto lam using degrade_spec()
    else:
        Teffs  = 5778.     # Sun effective temperature
        Rs  = 1.           # Sun radius (in solar radii)
        #Fsol  = Fstar(lam, Teffs, Rs, 1., AU=True)  # Sun as blackbody (W/m**2/um)
    rat   = np.zeros(len(lam))
    rat[:]= Fstar[:]/FsolV # ratio of solar flux to V-band solar flux
    if CIRC:
        # circular aperture size (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture size (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    return np.pi*q*T*Omega*dlam*(lam*1.e-6/hc)*(D/2.)**2.*(1./r)**2.*rat*Nez*F0V*10.**(-Mezv/2.5)

def cspeck(q, T, C, lam, dlam, Fstar, D):
    """
    Speckle count rate (Equation 19 from Robinson et al. 2016)

    :math:`{c}_{\\mathrm{sp}} = \\pi q{ \\mathcal T }C\\Delta \\lambda {F}_{{\\rm{s}},\\lambda }(d)\\displaystyle \\frac{\\lambda }{{hc}}{\\left(\\displaystyle \\frac{D}{2}\\right)}^{2}`

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    T : float
        System throughput
    C : float, optional
        Coronagraph design contrast
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    Fstar : float or array-like
        Host star spectrum at distance of planet (TOA) [W/m**2/um]

    Returns
    -------
    cspeck : float or array-like
        Speckle photon count rate [1/s]
    """
    hc    = 1.986446e-25 # h*c (kg*m**3./s**2.)
    return np.pi*q*T*C*dlam*Fstar*(lam*1.e-6/hc)*(D/2.)**2.

def cdark(De, X, lam, D, theta, DNhpix, IMAGE=False, CIRC=False):
    """
    Dark current photon count rate

    Parameters
    ----------
    De : float, optional
        Dark current [counts/s]
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    lam : float or array-like
        Wavelength [um]
    D : float
        Telescope diameter [m]
    theta :
        Angular size of lenslet or pixel [arcsec**2]
    DNHpix : float, optional
        Number of horizontal/spatial pixels for dispersed spectrum
    IMAGE : bool, optional
        Set to indicate imaging mode (not IFS)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    cdark :
        Dark current photon count rate (s**-1)
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
    return De*Npix

def cread(Re, X, lam, D, theta, DNhpix, Dtmax, IMAGE=False, CIRC=False):
    """
    Read noise count rate (assuming detector has a maximum exposure time)

    Parameters
    ----------
    Re : float or array-like
        Read noise counts per pixel
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    lam : float or array-like
        Wavelength [um]
    D : float
        Telescope diameter [m]
    theta :
        Angular size of lenslet or pixel [arcsec**2]
    Dtmax : float, optional
        Detector maximum exposure time [hours]
    IMAGE : bool, optional
        Set to indicate imaging mode (not IFS)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    cread : float or array-like
        Read noise photon count rate (s**-1)
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
    return Npix/(Dtmax*3600.)*Re

def ccic(Rc, cscene, X, lam, D, theta, DNhpix, Dtmax, IMAGE=False, CIRC=False):
    """
    Clock induced charge count rate

    Parameters
    ----------
    Rc : float or array-like
        Clock induced charge counts/pixel/photon
    cscene : float or array-like
        Photon count rate of brightest pixel in the scene [counts/s]
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    lam : float or array-like
        Wavelength [um]
    D : float
        Telescope diameter [m]
    theta :
        Angular size of lenslet or pixel [arcsec**2]
    Dtmax : float, optional
        Detector maximum exposure time [hours]
    IMAGE : bool, optional
        Set to indicate imaging mode (not IFS)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    ccic :
        Clock induced charge count rate [1/s]
    """
    # Convert from counts/pixel/photon --> counts/pixel/s
    Rcs = Rc * cscene
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
    return Npix/(Dtmax*3600.)*Rcs


def f_airy(X, CIRC=False):
    """
    Fraction of Airy power contained in square or circular aperture

    Parameters
    ----------
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    f_airy : float
        Fraction of planetary light that falls within photometric aperture X*lambda/D
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

'''
def f_airy_int(X, N = 1000):
    """
    Numerical integration to determine fraction of Airy power
    contained square aperture (SLOW!)

    Parameters
    ----------
    X : float, optional
        Width of photometric aperture ( * lambda / diam)

    Returns
    -------
    f_airy : float
        Fraction of planetary light that falls within photometric aperture X*lambda/D
    N : int
        Sets horizontal and vertical grid resolution
    """
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
'''

def ctherm(q, X, T, lam, dlam, D, Tsys, emis, CIRC=False):
    """
    Telescope thermal count rate

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    T : float
        System throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    Tsys  : float
        Telescope mirror temperature [K]
    emis : float
        Effective emissivity for the observing system (of order unity)
    CIRC : bool, optional
        Set to use a circular aperture


    Returns
    -------
    ctherm : float or array-like
        Telescope thermal photon count rate [1/s]
    """
    hc    = 1.986446e-25  # h*c (kg*m**3/s**2)
    c1    = 3.7417715e-16 # 2*pi*h*c*c (kg m**4 / s**3)
    c2    = 1.4387769e-2  # h*c/k (m K)
    lambd= 1.e-6*lam      # wavelength (m)
    power   = c2/lambd/Tsys
    Bsys  = c1/( (lambd**5.)*(np.exp(power)-1.) )*1.e-6/np.pi # system Planck function (W/m**2/um/sr)
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Omega = Omega/(206265.**2.) # aperture size (sr**2)
    return np.pi*q*T*dlam*emis*Bsys*Omega*(lam*1.e-6/hc)*(D/2)**2.

def ctherm_earth(q, X, T, lam, dlam, D, Itherm, CIRC=False):
    """
    Earth atmosphere thermal count rate

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    T : float
        System throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    Itherm : float or array-like
        Earth thermal intensity [W/m**2/um/sr]
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    cthe : float or array-like
        Earth atmosphere thermal photon count rate [1/s]
    """
    hc    = 1.986446e-25  # h*c (kg*m**3/s**2)
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Omega = Omega/(206265.**2.) # aperture size (sr**2)
    return np.pi*q*T*dlam*Itherm*Omega*(lam*1.e-6/hc)*(D/2)**2.

def lambertPhaseFunction(alpha):
    """
    Calculate the Lambertian Phase Function from the phase angle,

    :math:`{{\\rm{\\Phi }}}_{{\\rm{L}}}(\\alpha )=\\displaystyle \\frac{\\mathrm{sin}\\alpha +(\\pi -\\alpha )\\mathrm{cos}\\alpha }{\\pi }`

    Parameters
    ----------
    alpha: float
        Planet phase angle [deg]

    Returns
    -------
    Phi : float
        Lambertian phase function
    """
    alpha = alpha * np.pi / 180.
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi

def construct_lam(lammin, lammax, Res=None, dlam=None):
    """
    Construct a wavelength grid by specifying either a resolving power (`Res`)
    or a bandwidth (`dlam`)

    Parameters
    ----------
    lammin : float
        Minimum wavelength [microns]
    lammax : float
        Maximum wavelength [microns]
    Res : float, optional
        Resolving power (lambda / delta-lambda)
    dlam : float, optional
        Spectral element width for evenly spaced grid [microns]

    Returns
    -------
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    """

    # Keyword catching logic
    goR = False
    goL = False
    if ((Res is None) and (dlam is None)) or (Res is not None) and (dlam is not None):
        print("Error in construct_lam: Must specify either Res or dlam, but not both")
    elif Res is not None:
        goR = True
    elif dlam is not None:
        goL = True
    else:
        print("Error in construct_lam: Should not enter this else statment! :)")
        return None, None

    # If Res is provided, generate equal resolving power wavelength grid
    if goR:

        # Set wavelength grid
        dlam0 = lammin/Res
        dlam1 = lammax/Res
        lam  = lammin #in [um]
        Nlam = 1
        while (lam < lammax + dlam1):
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

        #Set edges to be same as neighbor
        dlam[0] = dlam0#dlam[1]
        dlam[Nlam-1] = dlam1#dlam[Nlam-2]

        lam = lam[:-1]
        dlam = dlam[:-1]

    # If dlam is provided, generate evenly spaced grid
    if goL:
        lam = np.arange(lammin, lammax+dlam, dlam)
        dlam = dlam + np.zeros_like(lam)

    return lam, dlam

def set_quantum_efficiency(lam, qe, NIR=False, qe_nir=0.9, vod=False):
    """
    Set instrumental quantum efficiency

    Parameters
    ----------
    lam : float or array-like
        Wavelength [um]
    qe : float
        Detector quantum efficiency
    NIR : bool, optional
        Use near-IR detector proporties
    q_nir : float, optional
        NIR quantum efficiency
    vod : bool, optional
        "Valley of Death" red QE parameterization from Robinson et al. (2016)

    Returns
    -------
    q : numpy.array
        Wavelength-dependent instrumental quantum efficiency
    """
    Nlam = len(lam)
    q = np.zeros(Nlam)
    for j in range(Nlam):
        if (lam[j] <= 0.7) or not vod:
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
    Set dark current grid as a function of wavelength

    Parameters
    ----------
    lam : array-like
        Wavelength grid [microns]
    De : float
        Dark current count rate per pixel (s**-1)
    lammax : float
        Maximum wavelength
    Tdet : float
        Detector Temperature [K]
    NIR : bool, optional
        Use near-IR detector proporties
    De_nir : float, optional
        NIR minimum dark current count rate per pixel

    Returns
    -------
    De : numpy.array
        Dark current as a function of wavelength
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
    Set read noise grid as a function of wavelength

    Parameters
    ----------
    lam : array-like
        Wavelength grid [microns]
    Re : float
        Read noise counts per pixel (s**-1)
    NIR : bool, optional
        Use near-IR detector proporties
    Re_nir : float, optional
        NIR read noise counts per pixel

    Returns
    -------
    Re : numpy.array
        Read noise as a function of wavelength
    """
    Re = np.zeros(len(lam)) + Re

    if NIR:
        iNIR  = (lam > 1.0)
        Re[iNIR] = Re_nir

    return Re

def set_lenslet(lam, lammin, diam, X,
                NIR=True, lammin_nir=1.0):
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
    X : float
        Width of photometric aperture (``*lambda/diam``)
    NIR : bool (optional)
        Use near-IR detector proporties
    lammin_nir : float (optional)
        Wavelength min to use for NIR lenslet size

    Returns
    -------
    theta : numpy.array
        Angular size of lenslet
    """
    Nlam = len(lam)
    if NIR:
        theta = np.zeros(Nlam)
        iVIS  = (lam <= lammin_nir)
        iNIR  = (lam > lammin_nir)
        theta[iVIS] = X*lammin/1e6/diam/2.*(180/np.pi*3600.)
        # If there are wavelength bins longer than 1um:
        theta[iNIR] = X*lammin_nir/1e6/diam/2.*(180/np.pi*3600.)
    else:
        # Should there be in X multiplying below?
        theta = X*lammin/1.e6/diam/2.*(180/np.pi*3600.) # assumes sampled at ~lambda/2D (arcsec)

    return theta

def set_throughput(lam, Tput, diam, sep, IWA, OWA, lammin,
                   FIX_OWA=False, SILENT=False):
    """
    Set wavelength-dependent telescope throughput such that it is zero
    inside the IWA and outside the OWA.

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
    SILENT : bool, optional
        Suppress printing

    Returns
    -------
    T : numpy.array
        Wavelength-dependent throughput
    """
    Nlam = len(lam)
    T    = Tput + np.zeros(Nlam)
    iIWA = ( sep < IWA*lam/diam/1.e6 )
    if (True if True in iIWA else False):
        T[iIWA] = 0. #zero transmission for points inside IWA have no throughput
        if not SILENT:
            print('WARNING: portions of spectrum inside IWA')
    if FIX_OWA:
        if ( sep > OWA*lammin/diam/1.e6 ):
            T[:] = 0. #planet outside OWA, where there is no throughput
            if not SILENT:
                print('WARNING: planet outside fixed OWA')
    else:
        iOWA = ( sep > OWA*lam/diam/1.e6 )
        if (True if True in iOWA else False):
            T[iOWA] = 0. #points outside OWA have no throughput
            if not SILENT:
                print('WARNING: portions of spectrum outside OWA')
    return T

def set_atmos_throughput(lam, dlam, convolve, plot=False):
    """
    Use pre-computed Earth atmospheric transmission to set throughput term for
    radiation through the atmosphere

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    dlam : ndarray
        Wavelength bin width grid
    convolve : func
        Function used to degrade/downbin spectrum

    Returns
    -------
    Tatmos : numpy.array
        Atmospheric throughput as a function of wavelength
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


def set_atmos_throughput_skyflux(wl_atmos, Tatmoshr, lam, dlam, convolve, plot=False):
    """
    Use SkyFlux Earth atmospheric transmission to set throughput term for
    radiation through the atmosphere

    Parameters
    ----------
    lam : ndarray
        Wavelength grid
    dlam : ndarray
        Wavelength bin width grid
    convolve : func
        Function used to degrade/downbin spectrum

    Returns
    -------
    Tatmos : numpy.array
        Atmospheric throughput as a function of wavelength
    """
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


'''
def get_thermal_ground_intensity(lam, dlam, convolve):
    """
    Get the intensity at the ground in each spectral element due to the sky
    background

    Parameters
    ----------
    lam : ndarray
        Wavelength grid [um]
    dlam : ndarray
        Wavelength bin width grid [um]
    convolve : func
        Function used to degrade/downbin spectrum

    Returns
    -------
    Itherm : ndarray
        Intensity at the ground due to the sky as a function of wavelength
        [W/m**2/um/sr]
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
'''

def get_sky_flux():
    """
    THIS FUNCTION WILL BE DEPRICIATED


    Get the spectral flux density from the sky viewed at a ground-based
    telescope an an average night. This calculation comes from
    `ESO SKYCALC <https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC>`_
    and includes contributions from molecular emission of lower atmosphere,
    emission lines of upper atmosphere, and airglow/residual continuum, but
    neglects scattered moonlight, starlight, and zodi.

    Returns
    -------
    lam_sky : `numpy.array`
        Wavelength grid [microns]
    flux_sky : `numpy.array`
        Flux from the sky [W/m^2/um]

    """
    hc    = 1.986446e-25 # h*c (kg*m**3/s**2)

    # Use ESO SKYCALC
    fn = os.path.join(os.path.dirname(__file__), "ground/skytable_no_zodi_moon.fits")
    df = fits.open(fn)
    lam_sky = df[1].data["lam"]
    flux_sky = df[1].data["flux"]
    trans_sky = df[1].data["trans"]

    # Convert sky flux to W/m^2/um to match existing function
    flux_sky = flux_sky * (np.pi * u.steradian)
    flux_sky = flux_sky * (u.radian.in_units(u.arcsec))**2 * (u.arcsec**2 / u.steradian)
    flux_sky = flux_sky / (lam_sky * 1e-6) * hc / (u.photon / u.watt / u.second)
    flux_sky = flux_sky.value / np.pi

    return lam_sky, flux_sky

def exptime_element(lam, cp, cn, wantsnr):
    """
    Calculate the exposure time (in hours) to get a specified signal-to-noise

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
    DtSNR[j] = np.nan
    return DtSNR

def planck(temp, wav):
    """
    Planck blackbody function

    Parameters
    ----------
    temp : float or array-like
        Temperature [K]
    wav : float or array-like
        Wavelength [microns]

    Returns
    -------
    B_lambda [W/m^2/um/sr]
    """
    h = 6.62607e-34       # Planck constant (J * s)
    c = 2.998e8           # Speed of light (m / s)
    k = 1.3807e-23        # Boltzmann constant (J / K)
    wav = wav * 1e-6
    # Returns B_lambda [W/m^2/um/sr]
    return 1e-6 * (2. * h * c**2) / (wav**5) / (np.exp(h * c / (wav * k * temp)) - 1.0)


def set_AO_fpa(X, AO_mode, D, ref_lam=0.76):
    """
    Sets the fraction of light captured by the aperture for ground-based
    observations.

    Parameters
    ----------
    theta : float or array-like
        Angular size of photometric aperture [arcsec]
    AO_mode : float or array-like
        Wavelength [microns]

    Returns
    -------
    A_fpa (frac)
    """

    if AO_mode is not None:
        if AO_mode == "seeing_limited":
            AO_fl = os.path.join(os.path.dirname(__file__), "ground/seeing_lim.txt")
        elif AO_mode == "ground_layer":
            AO_fl = os.path.join(os.path.dirname(__file__), "ground/GLAO.txt")
        elif AO_mode == "laser_tomography":
            AO_fl = os.path.join(os.path.dirname(__file__), "ground/LTAO.txt")

        R_ref = X * ref_lam / 1e6 / D / 2 * (180/np.pi*3600.) * 1000 # in mas
        band_central_lams = [0.36, 0.44, 0.55, 0.64, 0.79, 1.25, 1.65, 2.16, 3.50, 4.80, 10.20, 21.00]
        band_names = ["U", "B", "V", "R", "I", "J", "H", "K", "L", "M", "N", "Q"]
        closest_ind = min(range(len(band_central_lams)), key=lambda i: abs(band_central_lams[i]-ref_lam))
        band = band_names[closest_ind]

        data = pd.read_csv(AO_fl, sep=" ", header=0, comment='#')
        Rref_table, frac_light_table = data.Rref, data[band]/100 # data are listed as percentages, so divide by 100 to get frac
        interp_func = interp1d(Rref_table, frac_light_table, fill_value = "extrapolate")

        AO_fpa = interp_func(R_ref)
        return AO_fpa

    else:
        AO_fpa = 1.
        return AO_fpa

def calculate_Npix(X, lam, D, theta, DNhpix, IMAGE=False, CIRC=False):
    """
    Calculate number of pixels per wavelength element

    Parameters
    ----------
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    lam : float or array-like
        Wavelength [um]
    D : float
        Telescope diameter [m]
    theta :
        Angular size of lenslet or pixel [arcsec**2]
    DNHpix : float, optional
        Number of horizontal/spatial pixels for dispersed spectrum
    IMAGE : bool, optional
        Set to indicate imaging mode (not IFS)
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    Npix :
        number of pixels per wavelength element
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

    return Npix
