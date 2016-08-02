# Import dependent modules
import numpy as np
from .degrade_spec import degrade_spec
from .convolve_spec import convolve_spec
from .noise_routines import Fstar, Fplan, FpFs, cplan, czodi, cezodi, cspeck, cdark, cread, ctherm, ccic, f_airy
from .Noise import Output
import pdb

def count_rates_wrapper(Ahr, lamhr, solhr,
                   telescope, planet, star,
                   wantsnr=10.0, FIX_OWA = False, COMPUTE_LAM = False,
                   SILENT = False, NIR = True, THERMAL = False):
    """

    Parameters
    ----------
    Ahr : array
        hi-res planetary albedo spectrum
    lamhr : array
        wavelength grid for Ahr (um)
    solhr : array
        hi-res TOA solar spectrum (W/m**2/um)
    telescope : Telescope
        Telescope object containing parameters
    planet : Planet
        Planet object containing parameters
    star : Star
        Star object containing parameters
    FIX_OWA : bool
        set to fix OWA at OWA*lammin/D, as would occur if lenslet array is limiting the OWA
    COMPUTE_LAM : bool
        set to compute lo-res wavelength grid, otherwise the grid input as variable 'lam' is used
    NIR : bool
        re-adjusts pixel size in NIR, as would occur if a second instrument was designed to handle the NIR
    THERMAL : bool
        set to compute thermal photon counts due to telescope temperature
    """

    # Planet Parameters
    alpha  = planet.alpha           # Phase angle
    Phi    = planet.Phi             # Phase function
    Rp     = planet.Rp              # Planet radius (R_earth)
    r      = planet.a               # Semi-major axis (AU)
    d      = planet.distance        # Planet distance (pc)
    Nez    = planet.Nez             # Number of exo-zodi

    # Stellar Parameters
    Teff   = star.Teff              # Stellar effective temperature (K)
    Rs     = star.Rs                # Stellar radius (R_sun)

    # Telescope Parameters
    mode         = telescope.mode
    filter_wheel = telescope.filter_wheel
    lammin = telescope.lammin       # Wavelength minimum
    lammax = telescope.lammax       # Wavelength maximum
    Res    = telescope.resolution   # Resolving power
    diam   = telescope.diameter     # Diameter (m)
    Tput   = telescope.throughput   # Throughput
    C      = telescope.contrast     # Raw contrast
    IWA    = telescope.IWA          # Inner working angle
    OWA    = telescope.OWA          # Outer working angle
    Tsys   = telescope.Tsys         # Telescope/System temperature (K)
    Tdet   = telescope.Tdet         # Detector temperature (K)
    emis   = telescope.emissivity   # Emissivity

    # Set key system parameters
    De     = telescope.darkcurrent  # dark current (s**-1)
    DNHpix = telescope.DNHpix       # horizontal pixel spread of IFS spectrum
    Re     = telescope.readnoise    # read noise per pixel
    Dtmax  = telescope.Dtmax        # maximum exposure time (hr)
    X      = telescope.X            # size of photometric aperture (lambda/D)
    qe     = telescope.qe           # quantum efficiency

    # Set astrophysical parameters
    MzV  = 23.0                     # zodiacal light surface brightness (mag/arcsec**2)
    MezV = 22.0                     # exozodiacal light surface brightness (mag/arcsec**2)


    # Calculate count rates
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,
                    mode   = mode,
                    filter_wheel = filter_wheel,
                    lammin = lammin,
                    lammax = lammax,
                    Res    = Res,
                    diam   = diam,
                    Tput   = Tput,
                    C      = C,
                    IWA    = IWA,
                    OWA    = OWA,
                    Tsys   = Tsys,
                    Tdet   = Tdet,
                    emis   = emis,
                    De     = De,
                    DNHpix = DNHpix,
                    Re     = Re,
                    Dtmax  = Dtmax,
                    X      = X,
                    qe     = qe,
                    MzV    = MzV,
                    MezV   = MezV,
                    wantsnr=wantsnr, FIX_OWA=FIX_OWA, COMPUTE_LAM=COMPUTE_LAM,
                    SILENT=SILENT, NIR=NIR, THERMAL=THERMAL)

    # Cram all the coronagraph output arrays into an Output object
    output = Output(lam=lam, dlam=dlam, A=A, q=q, Cratio=Cratio,
                    cp=cp, csp=csp, cz=cz, cez=cez, cD=cD, cR=cR,
                    cth=cth, DtSNR=DtSNR)

    # Return output object
    return output


def count_rates(Ahr, lamhr, solhr,
                alpha, Phi, Rp, Teff, Rs, r, d, Nez,
                mode   = "IFS",
                filter_wheel = None,
                lammin = 0.4,
                lammax = 2.5,
                Res    = 70.0,
                diam   = 10.0,
                Tput   = 0.05,
                C      = 1e-10,
                IWA    = 3.0,
                OWA    = 20.0,
                Tsys   = 150.0,
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
                SILENT = False, NIR = True, THERMAL = False):
    """
    Generate photon count rates for specified telescope and planet parameters

    Parameters
    ----------
    Ahr : array
        hi-res planetary albedo spectrum
    lamhr : array
        wavelength grid for Ahr (um)
    solhr : array
        hi-res TOA solar spectrum (W/m**2/um)
    telescope : Telescope
        Telescope object containing parameters
    planet : Planet
        Planet object containing parameters
    star : Star
        Star object containing parameters
    FIX_OWA : bool
        set to fix OWA at OWA*lammin/D, as would occur if lenslet array is limiting the OWA
    COMPUTE_LAM : bool
        set to compute lo-res wavelength grid, otherwise the grid input as variable 'lam' is used
    NIR : bool
        re-adjusts pixel size in NIR, as would occur if a second instrument was designed to handle the NIR
    THERMAL : bool
        set to compute thermal photon counts due to telescope temperature
    """

    # Set the Imaging Mode?
    if mode == 'Imaging':
        filters = filter_wheel
        IMAGE = True
        COMPUTE_LAM = False
        # sorted filter dict by bandcenters
        tdict = sorted(filters.__dict__.iteritems(), key=lambda x: x[1].bandcenter)
        # Construct array of wavelengths
        lam = np.array([x[1].bandcenter for x in tdict])
        # Construct array of wavelength bin widths (FWHM)
        dlam = np.array([x[1].FWHM for x in tdict])
        Nlam = len(lam)
    else:
        IMAGE = False

    # fraction of planetary signal in Airy pattern
    fpa = f_airy(X)

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
        # Set wavelength widths
        for j in range(1,Nlam-1):
            dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])
        #widths at edges are same as neighbor
        dlam[0] = dlam[1]
        dlam[Nlam-1] = dlam[Nlam-2]
    elif IMAGE:
        pass
    else:
        # Throw error
        print "Error in make_noise: Not computing wavelength grid or providing filters!"
        return None

    # Set Quantum Efficiency
    q = np.zeros(Nlam)
    for j in range(Nlam):
        if (lam[j] <= 0.7):
            q[j] = qe
        else:
            q[j] = qe*(1.0 - (lam[j]-0.7)/(1.0-0.7))
    if q[j] < 0.0: q[j] = 0.

    # Set Dark current and Read noise
    De = np.zeros(Nlam) + De
    Re = np.zeros(Nlam) + Re

    # Set Angular size of lenslet
    theta = lammin/1.e6/diam/2.*(180/np.pi*3600.) # assumes sampled at ~lambda/2D (arcsec)
    if NIR:
        theta = np.zeros(Nlam)
        iVIS  = (lam <= 1.0)
        iNIR  = (lam > 1.0)
        theta[iVIS] = lammin/1e6/diam/2.*(180/np.pi*3600.)

        # If there are wavelength bins longer than 1um:
        if (np.sum(iNIR) > 0):
            theta[iNIR] = 1.0/1e6/diam/2.*(180/np.pi*3600.)
            q[iNIR]  = 0.90              # Different from vis detector
            Re[iNIR] = 2.                # Different from vis detector

            # Set dark current based on NIR detector properties
            if ( lammax <= 2.0 ): De[iNIR] = 1e-3 * np.power(10., (Tdet-120.)*7./100. )
            if ( lammax > 2.0 ) and ( lammax <= 4.0 ): De[iNIR] = 1e-3 * np.power(10., (Tdet-80.)*9./140. )
            if ( lammax > 4.0 ) and ( lammax <= 7.0 ): De[iNIR] = 1e-3 * np.power(10., (Tdet-40.)*11./140. )
            if ( lammax > 7.0 ): De[iNIR] = 1e-3 * np.power(10., (Tdet-30.)*11./70. )

            # Don't let dark current fall below a threshold
            iDe = (De[iNIR] < 1e-3)
            De[iDe] = 1e-3

    # Set throughput
    T    = Tput + np.zeros(Nlam)
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


    # Degrade albedo and stellar spectrum
    if COMPUTE_LAM:
        A = degrade_spec(Ahr,lamhr,lam,dlam=dlam)
        Fs = degrade_spec(solhr, lamhr, lam, dlam=dlam)
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
    cnoise =  cp + 2*(cz + cez + csp + cD + cR + cth)                        # assumes background subtraction
    cb = (cz + cez + csp + cD + cR + cth)
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
    DtSNR = np.zeros(Nlam)
    DtSNR[:] = 0.
    i = (cp > 0.)
    if (True if True in i else False):
        DtSNR[i] = (wantsnr**2.*cnoise[i])/cp[i]**2./3600. # (hr)

    return lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR
