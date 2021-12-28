"""
The crux of ``coronagraph`` noise modeling is to determine the photon count rate
incident upon the detector due to both the target planet and an assortment of
different telescope, instrumental, and astrophysical noise sources.
The following classes and functions serve as your interface to the photon count
rate calculations. The core function for these calculations is
:func:`count_rates`, but it may be accessed using the :class:`CoronagraphNoise`
object.
"""

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

# Import dependent modules
import astropy.units as u
import numpy as np
import sys, os
import matplotlib.pyplot as plt

from .degrade_spec import downbin_spec, doppler_shift
from .convolve_spec import convolve_spec
from .noise_routines import Fstar, Fplan, FpFs, cplan, czodi, cezodi, cspeck, \
    cdark, cread, ctherm, ccic, f_airy, ctherm_earth, construct_lam, \
    set_quantum_efficiency, set_read_noise, set_dark_current, set_lenslet, \
    set_throughput, set_atmos_throughput, set_atmos_throughput_skyflux, \
    exptime_element, get_sky_flux, cstar
from .teleplanstar import Telescope, Planet, Star

__all__ = ['count_rates', 'CoronagraphNoise']

class CoronagraphNoise(object):
    """
    The primary interface for ``coronagraph`` noise modeling. This object wraps
    around the functionality of :func:`count_rates`. Simply instantiate a
    `CoronagraphNoise` object by passing it :class:`telescope`, :class:`planet`,
    and :class:`star` objects, and then call
    :func:`CoronagraphNoise.run_count_rates` to perform the photon count rate
    calculation.

    Parameters
    ----------
    telescope : Telescope
        Initialized object containing ``Telescope`` parameters
    planet : Planet
        Initialized object containing ``Planet`` parameters
    star : Star
        Initialized object containing ``Star`` parameters
    skyflux : SkyFlux
        Initialized object containing ``SkyFlux`` parameters
    texp : float
        Exposure time for which to generate synthetic data [hours]
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
    ZODI : bool, optional
        Set to simulate zodiacal and exo-zodiacal photon noise
    vod : bool, optional
        "Valley of Death" red QE parameterization from Robinson et al. (2016)
    set_fpa : float, optional
        Specify the fraction of planetary signal in Airy pattern, default will
        calculate it from the photometric aperture size `X`
    roll_maneuver : bool, optional
        This assumes an extra factor of 2 hit to the background noise due to a
        telescope roll maneuver needed to subtract out the background. See
        Brown (2005) for more details.

    Note
    ----
    The results of the coronagraph noise calculation will become available as
    attributes of the :class:`CoronagraphNoise` object after
    :func:`CoronagraphNoise.run_count_rates` is called.
    """
    def __init__(self, telescope = Telescope(), planet = Planet(),
                 star = Star(), skyflux = None, texp = 10.0, wantsnr=10.0, FIX_OWA = False,
                 COMPUTE_LAM = False, SILENT = False, NIR = False,
                 THERMAL = True, GROUND = False, vod=False, set_fpa=None,
                 roll_maneuver = True, ZODI = True):
        """
        """
        self.telescope = telescope
        self.planet = planet
        self.star = star
        self.skyflux = skyflux
        self.texp = texp
        self.wantsnr = wantsnr
        self.FIX_OWA = FIX_OWA
        self.COMOUTE_LAM = COMPUTE_LAM
        self.SILENT = SILENT
        self.NIR = NIR
        self.THERMAL = THERMAL
        self.GROUND = GROUND
        self.vod = vod
        self.set_fpa = set_fpa
        self.roll_maneuver = roll_maneuver
        self.ZODI = ZODI
        self._computed = False

        return

    def run_count_rates(self, Ahr, lamhr, solhr):
        """
        Calculate the photon count rates and signal to noise on a
        coronagraph observation given a wavelength-dependent planetary
        geometric albedo and stellar flux density.

        Parameters
        ----------
        Ahr : array
            High-res, wavelength-dependent planetary geometric albedo
        lamhr : array
            High-res wavelength grid  [um]
        solhr : array
            High-res TOA solar spectrum [W/m**2/um]


        Calling ``run_count_rates()`` creates the following attributes for
        the ``CoronagraphNoise`` instance:

        Attributes
        ----------
        Ahr : array
            High-res, wavelength-dependent planetary geometric albedo
        lamhr : array
            High-res wavelength grid  [um]
        solhr : array
            High-res TOA solar spectrum [W/m**2/um]
        lam : array
            Observed wavelength grid [$\mu$m]
        dlam : array
            Observed wavelength grid widths [$\mu$m]
        A : array
            Planetary geometric albedo at observed resolution
        Cratio : array
            Planet-to-star flux contrast ratio
        cp : array
            Planetary photon count rate [photons/s]
        csp : array
            Speckle count rate [photons/s]
        cz : array
            Zodi photon count rate [photons/s]
        cez : array
            Exo-zodi photon count rate [photons/s]
        cth : array
            Thermal photon count rate [photons/s]
        cD : array
            Dark current photon count rate [photons/s]
        cR : array
            Read noise photon count rate [photons/s]
        cc : array
            Clock induced charge photon count rate [photons/s]
        cb : array
            Total background photon noise count rate [photons/s]
        DtSNR : array
            Integration time to ``wantsnr`` [hours]
        SNRt : array
            S/N in a ``texp`` hour exposure
        Aobs : array
            Observed albedo with noise
        Asig : array
            Observed uncertainties on albedo
        Cobs : array
            Observed Fp/Fs with noise
        Csig : array
            Observed uncertainties on Fp/Fs
        """

        # Save input arrays
        self.Ahr = Ahr
        self.lamhr = lamhr
        self.solhr = solhr

        # Aperture logic
        accepted_circular = ["circular", "circ", "c"]
        accepted_square = ["square", "s"]
        if self.telescope.aperture.lower() in accepted_circular:
            CIRC = True
        elif self.telescope.aperture.lower() in accepted_square:
            CIRC = False
        else:
            assert False, "telescope.aperture is invalid"

        # Call count_rates
        lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, cc, DtSNR, cs = \
            count_rates(Ahr, lamhr, solhr,
                        alpha = self.planet.alpha,
                        Phi = self.planet.Phi,
                        Rp = self.planet.Rp,
                        Teff = self.star.Teff,
                        Rs = self.star.Rs,
                        r = self.planet.a,
                        d = self.planet.distance,
                        Nez = self.planet.Nez,
                        mode = self.telescope.mode,
                        filter_wheel = self.telescope.filter_wheel,
                        lammin = self.telescope.lammin,
                        lammax = self.telescope.lammax,
                        Res    = self.telescope.resolution,
                        diam   = self.telescope.diameter,
                        Tput   = self.telescope.throughput,
                        C      = self.telescope.contrast,
                        IWA    = self.telescope.IWA,
                        OWA    = self.telescope.OWA,
                        Tsys   = self.telescope.Tsys,
                        Tdet   = self.telescope.Tdet,
                        emis   = self.telescope.emissivity,
                        De     = self.telescope.darkcurrent,
                        DNHpix = self.telescope.DNHpix,
                        Re     = self.telescope.readnoise,
                        Rc     = self.telescope.Rc,
                        Dtmax  = self.telescope.Dtmax,
                        X      = self.telescope.X,
                        qe     = self.telescope.qe,
                        MzV    = self.planet.MzV,
                        MezV   = self.planet.MezV,
                        A_collect = self.telescope.A_collect,
                        diam_circumscribed = self.telescope.diam_circumscribed,
                        diam_inscribed = self.telescope.diam_inscribed,
                        lam    = self.telescope.lam,
                        dlam   = self.telescope.dlam,
                        Tput_lam = self.telescope.Tput_lam,
                        qe_lam = self.telescope.qe_lam,
                        lammin_lenslet = self.telescope.lammin_lenslet,
                        NIR    = self.NIR,
                        GROUND = self.GROUND,
                        ZODI = self.ZODI
                        THERMAL = self.THERMAL,
                        CIRC = CIRC,
                        roll_maneuver = self.roll_maneuver,
                        SILENT = self.SILENT,
                        wantsnr = self.wantsnr,
                        skyflux = self.skyflux,
                        vs = self.star.vs,
                        vp = self.planet.vp,
                        vb = 0,
                    )

        # Save output arrays
        self.lam     = lam
        self.dlam    = dlam
        self.A       = A
        self.Cratio  = Cratio
        self.cp      = cp
        self.cs      = cs
        self.csp     = csp
        self.cz      = cz
        self.cez     = cez
        self.cD      = cD
        self.cR      = cR
        self.cth     = cth
        self.cc      = cc
        self.cb      = cz + cez + csp + cD + cR + cth + cc
        self.DtSNR   = DtSNR

        # Flip the switch
        self._computed = True

        # Make an initial set of fake data
        self.make_fake_data()

        return

    def make_fake_data(self, texp = None):
        """
        Make a fake/synthetic dataset by sampling from a Gaussian.

        Parameters
        ----------
        texp : float, optional
            Exposure time [hours]. If not provided, the ``CoronagraphNoise.texp``
            will be used by default.


        Calling ``make_fake_data()`` creates the following attributes for
        the ``CoronagraphNoise`` instance:

        Attributes
        ----------
        SNRt : array
            S/N in a ``texp`` hour exposure
        Aobs : array
            Observed albedo with noise
        Asig : array
            Observed uncertainties on albedo
        Cobs : array
            Observed Fp/Fs with noise
        Csig : array
            Observed uncertainties on Fp/Fs
        """

        # Ensure that simulation has been run
        assert self._computed

        # Allow new exposure time
        if texp is not None:
            self.texp = texp

        # Convert exposure time to seconds
        Dt = 3600. * self.texp

        # Use telescope roll maneuver
        if self.roll_maneuver:
            # assuming background subtraction (the "2")
            roll_factor = 2.0
        else:
            # standard background noise
            roll_factor = 1.0

        # Calculate signal-to-noise
        SNRt  = self.cp * Dt / np.sqrt((self.cp + roll_factor*self.cb) * Dt)
        SNRt_filtered = np.copy(SNRt)
        SNRt_floor = 1e-5 # introducing a floor to SNR measurements or else you get ridiculous observations
        SNRt_filter_inds = np.where(SNRt_filtered < SNRt_floor)
        SNRt_filtered[SNRt_filter_inds] = 0
        # Calculate 1-sigma errors on contrast ratio and albedo
        Csig = self.Cratio/SNRt_filtered
        Asig = self.A/SNRt_filtered

        # Calculate Gaussian noise
        gaus = np.random.randn(len(self.Cratio))

        # Add gaussian noise to observed data
        Cobs = self.Cratio + Csig * gaus
        Aobs = self.A + Asig * gaus

        # Save attributes
        self.SNRt = SNRt
        self.Asig = Asig
        self.Aobs = Aobs
        self.Csig = Csig
        self.Cobs = Cobs

        return

    def plot_spectrum(self, SNR_threshold = 1.0, Nsig = 6.0, ax0 = None,
                      err_kws = {"fmt" : ".", "c" : "k", "alpha" : 1},
                      plot_kws = {"lw" : 1.0, "c" : "C4", "alpha" : 0.5},
                      draw_box = True):
        """
        Plot noised direct-imaging spectrum.

        Parameters
        ----------
        SNR_threshold : float
            Threshold SNR below which do not plot
        Nsig : float
            Number of standard deviations about median observed points to set
            yaxis limits
        ax0 : `matplotlib.axes`
            Optional axis to provide
        err_kws : dic
            Keyword arguments for `errorbar`
        plot_kws : dic
            Keyword arguments for `plot`
        draw_box : bool
            Draw important quantities in a box?

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Returns a figure if `ax0` is `None`
        ax : `matplotlib.axes`
            Returns an axis if `ax0` is `None`

        Note
        ----
        Only returns `fig` and `ax` is ``ax0 is None``
        """

        m = [self.SNRt > SNR_threshold]

        scale = 1

        if ax0 is None:
            # Create Plot
            fig, ax = plt.subplots(figsize = (10,8))
            ax.set_xlabel(r"Wavelength [$\mu$m]")
            ax.set_ylabel("Geometic Albedo")
        else:
            ax = ax0

        #ax.plot(lam, scale*RpRs2, alpha = 1.0, ls = "steps-mid")
        ax.errorbar(self.lam[m], scale*self.Aobs[m], yerr=scale*self.Asig[m], zorder = 100, **err_kws)
        #ax.set_yscale("log")

        # Set ylim
        mederr = scale*np.nanmedian(self.Asig)
        medy = scale*np.nanmedian(self.Aobs)
        ax.set_ylim([medy - Nsig*mederr, medy + Nsig*mederr])

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        ax.plot(self.lamhr, scale*self.Ahr, **plot_kws)

        ax.set_ylim(ylims)
        ax.set_xlim(xlims)


        if draw_box:
            # Set string for plot text
            if self.texp > 2.0:
                timestr = "{:.0f}".format(self.texp)+' hours'
            else:
                timestr = "{:.0f}".format(self.texp*60)+' mins'
            plot_text = r'Distance = '+"{:.1f}".format(self.planet.distance)+' pc'+\
            '\n Integration time = '+timestr
            ax.text(0.02, 0.975, plot_text, transform=ax.transAxes, ha = "left", va = "top",
                    bbox=dict(boxstyle="square", fc="w", ec="k", alpha=0.9), zorder=101)

        #ax.legend()

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_SNR(self, ax0 = None, plot_kws = {"ls" : "steps-mid"}):
        """
        Plot the S/N on the planet as a function of wavelength.

        Parameters
        ----------
        ax0 : `matplotlib.axes`
            Optional axis to provide
        plot_kws : dic
            Keyword arguments for `plot`

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Returns a figure if `ax0` is `None`
        ax : `matplotlib.axes`
            Returns an axis if `ax0` is `None`

        Note
        ----
        Only returns `fig` and `ax` is ``ax0 is None``
        """
        if ax0 is None:
            # Create Plot
            fig, ax = plt.subplots(figsize = (10,8))
        else:
            ax = ax0

        ax.plot(self.lam, self.SNRt, **plot_kws)
        #ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("S/N on Planet Spectrum in %.1f hrs" %self.texp)
        #ax.legend()

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_time_to_wantsnr(self, ax0 = None, plot_kws = {"ls" : "steps-mid", "alpha" : 1.0}):
        """
        Plot the exposure time to get a SNR on the planet spectrum.

        Parameters
        ----------
        ax0 : `matplotlib.axes`
            Optional axis to provide
        plot_kws : dic
            Keyword arguments for `plot`

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Returns a figure if `ax0` is `None`
        ax : `matplotlib.axes`
            Returns an axis if `ax0` is `None`

        Note
        ----
        Only returns `fig` and `ax` is ``ax0 is None``
        """

        if ax0 is None:
            # Create Plot
            fig, ax = plt.subplots(figsize = (10,8))
            ax.set_xlabel(r"Wavelength [$\mu$m]")
            ax.set_ylabel("Hours to S/N = %i on Planet Spectrum" %self.wantsnr)
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.DtSNR, **plot_kws)

        if ax0 is None:
            return fig, ax
        else:
            return

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
                Rc     = 0.0,
                Dtmax  = 1.0,
                X      = 1.5,
                qe     = 0.9,
                MzV    = 23.0,
                MezV   = 22.0,
                A_collect = None,
                diam_circumscribed = None,
                diam_inscribed = None,
                lam    = None,
                dlam   = None,
                Tput_lam = None,
                qe_lam = None,
                lammin_lenslet = None,
                wantsnr=10.0, FIX_OWA = False, COMPUTE_LAM = False,
                SILENT = False, NIR = False, THERMAL = False, GROUND = False, ZODI=True
                vod=False, set_fpa=None, CIRC = True, roll_maneuver = True, skyflux=None, vs=0, vp=0, vb=0):
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
    Rc : float, optional
        Clock induced charge [counts/pixel/photon]
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
    A_collect : float, optional
        Mirror collecting area (m**2) (uses :math:`\pi(D/2)^2` by default)
    diam_circumscribed : float, optional
        Circumscribed telescope diameter [m] used for IWA and OWA (uses `diam`
        if `None` provided)
    diam_inscribed : float, optional
        Inscribed telescope diameter [m] used for lenslet calculations
        (uses `diam` if `None` provided)
    lam : array-like, optional
        Wavelength grid for spectrograph [microns] (uses ``lammin``, ``lammax``,
        and ``resolution`` to determine if ``None`` provided)
    dlam : array-like, optional
        Wavelength grid `widths` for spectrograph [microns] (uses ``lammin``, ``lammax``,
        and ``resolution`` to determine if ``None`` provided)
    Tput_lam : tuple of arrays
        Wavelength-dependent throughput e.g. ``(wls, tputs)``
    qe_lam : tuple of arrays
        Wavelength-dependent throughput e.g. ``(wls, qe)``
    lammin_lenslet : float, optional
        Minimum wavelength to use for lenslet calculation (default is ``lammin``)
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
    CIRC : bool, optional
        Set to use a circular aperture
    roll_maneuver : bool, optional
        This assumes an extra factor of 2 hit to the background noise due to a
        telescope roll maneuver needed to subtract out the background. See
        Brown (2005) for more details.
    skyflux : SkyFlux
        Initialized object containing ``SkyFlux`` parameters
    vs : float
        stellar radial velocity in km/s
    vp : float
        planetary radial velocity relative to star in km/s
    vb : float
        barycentric radial velocity in km/s relative to star

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
        Planetary photon count rate on detector [1/s]
    csp : ndarray
        Speckle photon count rate on detector [1/s]
    cz : ndarray
        Zodiacal photon count rate on detector [1/s]
    cez : ndarray
        Exozodiacal photon count rate on detector [1/s]
    cD : ndarray
        Dark current photon count rate on detector [1/s]
    cR : ndarray
        Read noise photon count rate on detector [1/s]
    cth : ndarray
        Instrument thermal photon count rate on detector [1/s]
    cc : ndarray
        Clock induced charge photon count rate [1/s]
    DtSNR : ndarray
        Exposure time required to get desired S/N (wantsnr) [hours]
    """

    convolution_function = downbin_spec
    #convolution_function = degrade_spec

    # Define a diameter for IWA (circumscribed),
    # collecting area, and lenslet (inscribed)
    if diam_inscribed is None:
        # Defaults to diam
        diam_inscribed = diam
    if A_collect is None:
        # Defaults to diam
        diam_collect = diam
    else:
        # Calculated from provided collecting area
        diam_collect = 2. * (A_collect / np.pi)**0.5
    if diam_circumscribed is None:
        # Defaults to diam
        diam_circumscribed = diam

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
    # GENERALIZE THIS:
    if COMPUTE_LAM:
        if (lam is None) or (dlam is None):
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
    if lammin_lenslet is None: lammin_lenslet = lammin
    theta = set_lenslet(lam, lammin_lenslet, diam_inscribed, X, NIR=True)

    # Set throughput (for inner and outer working angle cutoffs)
    sep  = r/d*np.sin(alpha*np.pi/180.)*np.pi/180./3600. # separation in radians
    T = set_throughput(lam, Tput, diam_circumscribed, sep, IWA, OWA, lammin, FIX_OWA=FIX_OWA, SILENT=SILENT)

    # Apply wavelength-dependent throuput, if needed
    if Tput_lam is not None:
        # Bin input throughput curve to native res
        Tlam = np.interp(lam, Tput_lam[0], Tput_lam[1])
        # Multiply into regular throughput
        T = T * Tlam

    # Apply wavelength-dependent quantum efficiency, if needed
    if qe_lam is not None:
        # Bin input QE curve to native res
        qlam = np.interp(lam, qe_lam[0], qe_lam[1])
        # Multiply into regular QE
        q = q * qlam

    # Modify throughput by atmospheric transmission if GROUND-based
    if GROUND:
        #if GROUND == "ESO":
            # Use ESO SKYCALC
        #    pass
        #else:
        # Use SMART calc
        Tatmos = set_atmos_throughput_skyflux(skyflux.lam, skyflux.trans, lam, dlam, convolution_function)
        # Multiply telescope throughput by atmospheric throughput
        #T = T * Tatmos
    else:
        Tatmos = np.ones_like(lam)

    # Degrade albedo and stellar spectrum
    if COMPUTE_LAM:
        A = convolution_function(Ahr, lamhr, lam, dlam=dlam)
        Fs = convolution_function(solhr, lamhr, lam, dlam=dlam) # stellar flux at planet TOA
    elif IMAGE:
        # Convolve with filter response
        A = convolve_spec(Ahr, lamhr, filters)
        Fs = convolve_spec(solhr, lamhr, filters)
    else:
        A = Ahr
        Fs = solhr

    # Compute fluxes
    Bstar = Fs / ( np.pi*(Rs*u.Rsun.in_units(u.km)/\
                   (r*u.AU.in_units(u.km)))**2. )
    omega_star = np.pi*(Rs*u.Rsun.in_units(u.km)/\
                       (d*u.pc.in_units(u.km)))**2.
    Fs_earth = Bstar * omega_star # stellar flux at earth

    Fp = Fplan(A, Phi, Fs, Rp, d)         # planet flux at telescope; don't doppler shift the star here
    Cratio = FpFs(A, Phi, Rp, r)

    # # doppler shift Fp and Cratio to total planet RV
    # Fp = doppler_shift(lam, Fp, vs+vp+vb)
    # Cratio = doppler_shift(lam, Cratio, vs+vp+vb)
    #
    # # now we can doppler shift the star to total stellar RV
    # Fs_earth = doppler_shift(lam, Fs_earth, vs+vb)

    T2 = T * Tatmos # two-component throughput (Tatmos not 1 for ground)

    ##### Compute count rates #####
    cp     =  cplan(q, fpa, T2, lam, dlam, Fp, diam_collect)                          # planet count rate
    cs     =  cstar(q, fpa, T2, lam, dlam, Fs_earth, diam_collect)
    if ZODI:
        cz     =  czodi(q, X, T2, lam, dlam, diam_collect, MzV)                           # solar system zodi count rate
        cez    =  cezodi(q, X, T2, lam, dlam, diam_collect, r, \
                        Fs_earth, Nez, MezV)                                    # exo-zodi count rate
    else:
        cz = np.zeros_like(cs)
        cez = np.zeros_like(cs)
    csp    =  cspeck(q, T2, C, lam, dlam, Fs_earth, diam_collect)         # speckle count rate
    cD     =  cdark(De, X, lam, diam_collect, theta, DNHpix, IMAGE=IMAGE)            # dark current count rate
    cR     =  cread(Re, X, lam, diam_collect, theta, DNHpix, Dtmax, IMAGE=IMAGE)     # readnoise count rate
    if THERMAL:
        cth    =  ctherm(q, X, T, lam, dlam, diam_collect, Tsys, emis)               # internal thermal count rate
    else:
        cth = np.zeros_like(cp)

    # Add earth thermal photons if GROUND
    if skyflux is not None:
        if GROUND == "ESO":
            # Use ESO SKCALC
            wl_sky, Isky = get_sky_flux()
            # Convolve to instrument resolution
            Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)

        elif GROUND == "SKYFLUX":
            # Custom ESO SkyCalc option. See sky_flux.py. Must pass in a skyflux object
            wl_sky = skyflux.lam
            Isky = skyflux.flux
            # Convolve to instrument resolution
            Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)

        # Compute Earth thermal photon count rate
        cthe = ctherm_earth(q, X, T, lam, dlam, diam_collect, Itherm)
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

    # Clock induced charge photon count rate
    # Calculate photon count rate in the scene (everything except read noise)
    cscene = cp + cz + cez + csp + cD + cth
    # Calculate the clock induced charge photon count rate
    cc = ccic(Rc, cscene, X, lam, diam_collect, theta, DNHpix, Dtmax,
              IMAGE=IMAGE, CIRC=CIRC)

    # Calculate total background counts
    cb = (cz + cez + csp + cD + cR + cth + cc)

    # Use telescope roll maneuver
    if roll_maneuver:
        # assuming background subtraction (the "2")
        roll_factor = 2.0
    else:
        # standard background noise
        roll_factor = 1.0

    # Calculate total noise
    cnoise =  cp + roll_factor*cb

    # Calculate total counts
    ctot = cp + cz + cez + csp + cD + cR + cth + cc

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

    return lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, cc, DtSNR, cs
