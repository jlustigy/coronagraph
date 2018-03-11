from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

# Import dependent modules
import numpy as np
import sys
from .degrade_spec import degrade_spec, downbin_spec
from .convolve_spec import convolve_spec
from .noise_routines import Fstar, Fplan, FpFs, cplan, czodi, cezodi, cspeck, \
    cdark, cread, ctherm, ccic, f_airy, ctherm_earth, construct_lam, \
    set_quantum_efficiency, set_read_noise, set_dark_current, set_lenslet, \
    set_throughput, set_atmos_throughput, \
    exptime_element, get_sky_flux
import pdb
import os

__all__ = ['count_rates']

def count_rates(Ahr, lamhr, solhr,
                alpha, Phi, Rp, Teff, Rs, r, d, Nez,
                mode   = "IFS",
                filter_wheel = None,
                lammin = 0.4,
                lammax = 2.5,
                Res    = 70.0,
                diam   = 10.0,
                Tput   = 0.20,
                C      = 1e-10,
                IWA    = 3.0,
                OWA    = 20.0,
                Tsys   = 260.0,
                Tdet   = 50.0,
                emis   = 0.9,
                De     = 1e-4,
                DNHpix = 3.0,
                Re     = 0.1,
                Dtmax  = 1.0,
                X      = 1.5,
                qe     = 0.9,
                MzV    = 23.0,
                MezV   = 22.0,
                wantsnr=10.0, FIX_OWA = False, COMPUTE_LAM = False,
                SILENT = False, NIR = True, THERMAL = False, GROUND = False,
                vod=False, set_fpa=None):
    """
    Runs coronagraph model (Robinson et al., 2016) to calculate planet and noise
    photon count rates for specified telescope and system parameters.

    Parameters
    ----------
    Ahr : array
        High-res, wavelength-dependent planetary geometric albedo
    lamhr : array
        High-res wavelength grid  [um]
    solhr : array
        High-res TOA solar spectrum [W/m**2/um]
    alpha : float
        Planet phase angle [deg]
    Phi : float
        Planet phase function
    Rp : float
        Planet radius [R_earth]
    Teff : float
        Stellar effective temperature [K]
    Rs : float
        Stellar radius [R_sun]
    r : float
        Planet semi-major axis [AU]
    d : float
        Distance to observed star-planet system [pc]
    Nez : float
        Number of exozodis in exoplanetary disk
    mode : str, optional
        Telescope observing mode: "IFS" or "Imaging"
    filter_wheel : Wheel, optional
        Wheel object containing imaging filters
    lammin : float, optional
        Minimum wavelength [um]
    lammax : float, optional
        Maximum wavelength [um]
    Res : float, optional
        Instrument spectral resolution (``lam / dlam``)
    diam : float, optional
        Telescope diameter [m]
    Tput : float, optional
        Telescope and instrument throughput
    C : float, optional
        Coronagraph design contrast
    IWA : float, optional
        Coronagraph Inner Working Angle (``lam / diam``)
    OWA : float, optional
        Coronagraph Outer Working Angle (``lam / diam``)
    Tsys  : float, optional
        Telescope mirror temperature [K]
    Tdet  : float, optional
        Telescope detector temperature [K]
    emis : float, optional
        Effective emissivity for the observing system (of order unity)
    De : float, optional
        Dark current [counts/s]
    DNHpix : float, optional
        Number of horizontal/spatial pixels for dispersed spectrum
    Re : float, optional
        Read noise counts per pixel
    Dtmax : float, optional
        Detector maximum exposure time [hours]
    X : float, optional
        Width of photometric aperture (``lam / diam``)
    qe : float, optional
        Detector quantum efficiency
    MzV : float, optional
        V-band zodiacal light surface brightness [mag/arcsec**2]
    MezV : float, optional
        V-band exozodiacal light surface brightness [mag/arcsec**2]
    wantsnr : float, optional
        Desired signal-to-noise ratio in each pixel
    FIX_OWA : bool, optional
        Set to fix OWA at ``OWA*lammin/D``, as would occur if lenslet array is
        limiting the OWA
    COMPUTE_LAM : bool, optional
        Set to compute lo-res wavelength grid, otherwise the grid input as
        variable ``lam`` is used
    SILENT : bool, optional
        Set to suppress print statements
    NIR : bool, optional
        Re-adjusts pixel size in NIR, as would occur if a second instrument
        was designed to handle the NIR
    THERMAL : bool, optional
        Set to compute thermal photon counts due to telescope temperature
    GROUND : bool, optional
        Set to simulate ground-based observations through atmosphere
    vod : bool, optional
        "Valley of Death" red QE parameterization from Robinson et al. (2016)
    set_fpa : float, optional
        Specify the fraction of planetary signal in Airy pattern, default will
        calculate it from the photometric aperture size `X`

    Returns
    -------
    lam : ndarray
        Observational wavelength grid [um]
    dlam : ndarray
        Observational spectral element width [um]
    A : ndarray
        Planetary geometric albedo on lam grid
    q : ndarray
        Quantum efficiency grid
    Cratio : ndarray
        Planet-star contrast ratio
    cp : ndarray
        Planetary photon count rate on detector
    csp : ndarray
        Speckle photon count rate on detector
    cz : ndarray
        Zodiacal photon count rate on detector
    cez : ndarray
        Exozodiacal photon count rate on detector
    cD : ndarray
        Dark current photon count rate on detector
    cR : ndarray
        Read noise photon count rate on detector
    cth : ndarray
        Instrument thermal photon count rate on detector
    DtSNR : ndarray
        Exposure time required to get desired S/N (wantsnr) [hours]
    """

    convolution_function = downbin_spec
    #convolution_function = degrade_spec

    # Configure for different telescope observing modes
    if mode == 'Imaging':
        filters = filter_wheel
        IMAGE = True
        COMPUTE_LAM = False
        # sorted filter dict by bandcenters
        tdict = sorted(filters.__dict__.items(), key=lambda x: x[1].bandcenter)
        # Construct array of wavelengths
        lam = np.array([x[1].bandcenter for x in tdict])
        # Construct array of wavelength bin widths (FWHM)
        dlam = np.array([x[1].FWHM for x in tdict])
        Nlam = len(lam)
    elif mode == 'IFS':
        IMAGE = False
        COMPUTE_LAM = True
    else:
        print("Invalid telescope observing mode. Select 'IFS', or 'Imaging'.")
        sys.exit()

    # fraction of planetary signal in Airy pattern
    if set_fpa is None:
        fpa = f_airy(X)
    else:
        fpa = set_fpa * f_airy(X)

    # Set wavelength grid
    if COMPUTE_LAM:
        lam, dlam = construct_lam(lammin, lammax, Res)
    elif IMAGE:
        pass
    else:
        # Throw error
        print("Error in make_noise: Not computing wavelength grid or providing filters!")
        return None

    # Set Quantum Efficiency
    q = set_quantum_efficiency(lam, qe, NIR=NIR, vod=vod)

    # Set Dark current and Read noise
    De = set_dark_current(lam, De, lammax, Tdet, NIR=NIR)
    Re = set_read_noise(lam, Re, NIR=NIR)

    # Set Angular size of lenslet
    theta = set_lenslet(lam, lammin, diam, X, NIR=NIR)

    # Set throughput
    sep  = r/d*np.sin(alpha*np.pi/180.)*np.pi/180./3600. # separation in radians
    T = set_throughput(lam, Tput, diam, sep, IWA, OWA, lammin, FIX_OWA=FIX_OWA, SILENT=SILENT)

    # Modify throughput by atmospheric transmission if GROUND-based
    if GROUND:
        #if GROUND == "ESO":
            # Use ESO SKYCALC
        #    pass
        #else:
        # Use SMART calc
        Tatmos = set_atmos_throughput(lam, dlam, convolution_function)
        # Multiply telescope throughput by atmospheric throughput
        T = T * Tatmos

    # Degrade albedo and stellar spectrum
    if COMPUTE_LAM:
        A = convolution_function(Ahr,lamhr,lam,dlam=dlam)
        Fs = convolution_function(solhr, lamhr, lam, dlam=dlam)
    elif IMAGE:
        # Convolve with filter response
        A = convolve_spec(Ahr, lamhr, filters)
        Fs = convolve_spec(solhr, lamhr, filters)
    else:
        A = Ahr
        Fs = solhr

    # Compute fluxes
    #Fs = Fstar(lam, Teff, Rs, r, AU=True) # stellar flux on planet
    Fp = Fplan(A, Phi, Fs, Rp, d)         # planet flux at telescope
    Cratio = FpFs(A, Phi, Rp, r)

    ##### Compute count rates #####
    cp     =  cplan(q, fpa, T, lam, dlam, Fp, diam)                            # planet count rate
    cz     =  czodi(q, X, T, lam, dlam, diam, MzV)                           # solar system zodi count rate
    cez    =  cezodi(q, X, T, lam, dlam, diam, r, \
        Fstar(lam,Teff,Rs,1.,AU=True), Nez, MezV)                            # exo-zodi count rate
    csp    =  cspeck(q, T, C, lam, dlam, Fstar(lam,Teff,Rs,d), diam)         # speckle count rate
    cD     =  cdark(De, X, lam, diam, theta, DNHpix, IMAGE=IMAGE)            # dark current count rate
    cR     =  cread(Re, X, lam, diam, theta, DNHpix, Dtmax, IMAGE=IMAGE)     # readnoise count rate
    if THERMAL:
        cth    =  ctherm(q, X, lam, dlam, diam, Tsys, emis)                      # internal thermal count rate
    else:
        cth = np.zeros_like(cp)

    # Add earth thermal photons if GROUND
    if GROUND:
        # Use ESO SKCALC
        wl_sky, Isky = get_sky_flux()
        # Convolve to instrument resolution
        Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)
        # Compute Earth thermal photon count rate
        cthe = ctherm_earth(q, X, lam, dlam, diam, Itherm)
        # Add earth thermal photon counts to telescope thermal counts
        cth = cth + cthe
        '''
        if True:
            import matplotlib.pyplot as plt;
            fig2, ax1 = plt.subplots(figsize=(8,6))
            ax1.plot(lam, cthe, c="blue", ls="steps-mid", label="Earth Thermal")
            ax1.plot(lam, cth, c="red", ls="steps-mid", label="Telescope Thermal")
            ax1.plot(lam, cp, c="k", ls="steps-mid", label="Planet")
            ax1.set_ylabel("Photon Count Rate [1/s]")
            ax1.set_xlabel("Wavelength [um]")
            ax1.legend()
            plt.show()
        '''

    cb = (cz + cez + csp + cD + cR + cth)
    cnoise =  cp + 2*cb                # assumes background subtraction
    ctot = cp + cz + cez + csp + cD + cR + cth

    '''
    Giada: where does the factor of 2 come from [above]?

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
    DtSNR = exptime_element(lam, cp, cnoise, wantsnr)

    return lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR
