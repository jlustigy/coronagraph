#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate exoplanet transmission and/or emission spectroscopy without
using the coronagraph routines. This uses the same telesope and detector
parameters as the coronagraph model, but does not suppress the star's light.
As a result, stellar photons dominate the noise budget.

For transmission spectroscopy calculations use :class:`TransitNoise`,
and for emission spectroscopy use :class:`EclipseNoise`. You may also get an
example transmission and emission spectrum of the Earth by calling
:func:`get_earth_trans_spectrum`.

"""

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import sys, os

from .noise_routines import *
from .degrade_spec import *
from .observe import random_draw
from .teleplanstar import *
from .sky_flux import *
from .noise_routines import set_atmos_throughput_skyflux

__all__ = ["TransitNoise", "EclipseNoise", "get_earth_trans_spectrum"]

h = 6.62607004e-34
c = 2.998e8

class EclipseNoise(object):
    """
    Simulate exoplanet secondary eclipse emission spectroscopy with a next-generation
    telescope.

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
    tdur : float
        Transit duration [s]
    ntran : float
        Number of transits/eclipses
    nout : float
        Number of out-of-eclipse transit durations to observe
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

    """
    def __init__(self, tdur    = 3432.,  # TRAPPIST-1e
                       telescope = Telescope(),
                       planet = Planet(),
                       star = Star(),
                       skyflux = None,
                       ntran   = 1,
                       nout    = 1,
                       wantsnr = 1000.0,
                       NIR     = True,
                       THERMAL = True,
                       GROUND  = False,
                       vod     = False,
                       IMAGE   = False):
        self.telescope = telescope
        self.planet    = planet
        self.star      = star
        self.skyflux   = skyflux
        self.tdur      = tdur
        self.ntran     = ntran
        self.nout      = nout
        self.wantsnr   = wantsnr
        self.NIR       = NIR
        self.THERMAL   = THERMAL
        self.GROUND    = GROUND
        self.vod       = vod
        self.IMAGE     = IMAGE

        self._computed = False

        return

    def run_count_rates(self, lamhr = None, Fphr = None, Fshr = None):
        """
        Calculate the photon count rates and signal to noise on a secondary
        eclipse spectrum observation

        Parameters
        ----------
        lamhr : numpy.ndarray
            Wavelength [$\mu$m]
        Fphr : numpy.ndarray
            Dayside exoplanet TOA flux spectrum [W/m$^2$/$\mu$]
        Fshr : numpy.ndarray
            Stellar flux incident at the planet's TOA [W/m$^2$/$\mu$]

        Calling ``run_count_rates()`` creates the following attributes for
        the ``EclipseNoise`` instance:

        Attributes
        ----------
        lamhr : array
            Wavelength [$\mu$m]
        Fphr : array
            Dayside exoplanet TOA flux spectrum [W/m$^2$/$\mu$]
        Fshr : array
            Stellar flux incident at the planet's TOA [W/m$^2$/$\mu$]
        cs : array
            Stellar photon count rate [photons/s]
        cback : array
            Background photon count rate [photons/s]
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
        cmiss : array
            Occulted stellar photon count rate [photons/s]
        SNR1 : array
            S/N for one eclipse
        SNRn : array
            S/N for ``ntran`` eclipses
        tSNR : array
            Exposure time to ``wantsnr`` [s]
        nSNR : array
            Number of eclipses to ``wantsnr``
        lam : array
            Observed wavelength grid [$\mu$m]
        dlam : array
            Observed wavelength grid widths [$\mu$m]
        FpFslr : array
            Low-res planet/star flux ratio
        FpFshr : array
            High-res planetr/star flux ratio
        """

        self.lamhr = lamhr
        self.Fphr = Fphr
        self.Fshr = Fshr

        if self.telescope.A_collect is None:
            diam_collect = self.telescope.diameter
        else:
            diam_collect = 2. * (self.telescope.A_collect / np.pi)**0.5

        # Set the convolution function
        convolution_function = downbin_spec

        # Does the telescope object already have a wavelength grid?
        if (self.telescope.lam is None) or (self.telescope.dlam is None):
            # Create wavelength grid
            lam, dlam = construct_lam(self.telescope.lammin,
                                      self.telescope.lammax,
                                      self.telescope.resolution)
        else:
            # Use existing grids
            lam = self.telescope.lam
            dlam = self.telescope.dlam

        # Set Quantum Efficiency
        q = set_quantum_efficiency(lam,
                                   self.telescope.qe,
                                   NIR=self.NIR,
                                   vod=self.vod)

        # Set Dark current and Read noise
        De = set_dark_current(lam,
                              self.telescope.darkcurrent,
                              self.telescope.lammax,
                              self.telescope.Tdet,
                              NIR=self.NIR)
        Re = set_read_noise(lam,
                            self.telescope.readnoise,
                            NIR=self.NIR)

        # Set Angular size of lenslet
        theta = set_lenslet(lam,
                            self.telescope.lammin,
                            diam_collect,
                            self.telescope.X,
                            NIR=self.NIR)

        # Set throughput
        #sep  = r/d*np.sin(alpha*np.pi/180.)*np.pi/180./3600. # separation in radians
        #T = set_throughput(lam, Tput, diam, sep, IWA, OWA, lammin, FIX_OWA=FIX_OWA, SILENT=SILENT)
        T = self.telescope.throughput * np.ones_like(lam)

        # Apply wavelength-dependent throuput, if needed
        if self.telescope.Tput_lam is not None:
            # Bin input throughput curve to native res
            Tlam = np.interp(lam, self.telescope.Tput_lam[0], self.telescope.Tput_lam[1])
            # Multiply into regular throughput
            T = T * Tlam

        # Apply wavelength-dependent quantum efficiency, if needed
        if self.telescope.qe_lam is not None:
            # Bin input QE curve to native res
            qlam = np.interp(lam, self.telescope.qe_lam[0], self.telescope.qe_lam[1])
            # Multiply into regular QE
            q = q * qlam

        # Modify throughput by atmospheric transmission if GROUND-based
        if self.GROUND:
            # Use SMART calc
            #Tatmos = set_atmos_throughput(lam, dlam, convolution_function)
            Tatmos = set_atmos_throughput_skyflux(self.skyflux.lam, self.skyflux.trans, lam, dlam, convolution_function)
            # Multiply telescope throughput by atmospheric throughput
            T = T * Tatmos

        # Calculate intensity of the planet [W/m^2/um/sr]
        if Fphr is None:
            # Using a blackbody
            Bplan = planck(self.planet.Tplan, lamhr)
        else:
            # Using provided TOA planet flux
            Bplan = Fphr / np.pi

        # Calculate intensity of the star [W/m^2/um/sr]
        if Fshr is None:
            # Using a blackbody
            Bstar = planck(self.star.Teff, lamhr)
        else:
            # Using provided TOA stellar flux
            Bstar = Fshr / ( np.pi*(self.star.Rs*u.Rsun.in_units(u.km)/\
                           (self.planet.a*u.AU.in_units(u.km)))**2. )

        # Solid angle in steradians
        omega_star = np.pi*(self.star.Rs*u.Rsun.in_units(u.km)/\
                           (self.planet.distance*u.pc.in_units(u.km)))**2.
        omega_planet = np.pi*(self.planet.Rp*u.Rearth.in_units(u.km)/\
                             (self.planet.distance*u.pc.in_units(u.km)))**2.

        # Fluxes at earth [W/m^2/um]
        Fs = Bstar * omega_star
        Fp = Bplan * omega_planet
        FpFs = Fp/Fs

        # Degrade planet and stellar spectrum to instrument res
        Fplr = convolution_function(Fp, lamhr, lam, dlam=dlam)
        Fslr = convolution_function(Fs, lamhr, lam, dlam=dlam)
        FpFslr = convolution_function(FpFs, lamhr, lam, dlam=dlam)

        # Fraction of planetary signal in Airy pattern
        fpa = 1.0   # No fringe pattern here --> all of stellar psf falls on CCD

        ########## Calculate Photon Count Rates ##########

        # Planet photon count rate
        cp = cplan(q, fpa, T, lam, dlam, Fplr, diam_collect)

        # Stellar photon count rate
        cs = cstar(q, fpa, T, lam, dlam, Fslr, diam_collect)

        # Solar System Zodi count rate
        cz =  czodi(q, self.telescope.X, T, lam, dlam,
                    diam_collect, self.planet.MzV)

        # Exo-Zodi count rate
        cez =  cezodi(q, self.telescope.X, T, lam, dlam, diam_collect,
                      self.planet.a,
                      Fstar(lam, self.star.Teff, self.star.Rs, 1., AU=True),
                      self.planet.Nez, self.planet.MezV)

        # Dark current count rate
        cD =  cdark(De, self.telescope.X, lam,
                    diam_collect, theta,
                    self.telescope.DNHpix, IMAGE=self.IMAGE)

        # Read noise count rate
        cR =  cread(Re, self.telescope.X, lam, diam_collect,
                    theta, self.telescope.DNHpix, self.telescope.Dtmax,
                    IMAGE=self.IMAGE)

        # Thermal background count rate
        if self.THERMAL:
            # telescope internal thermal count rate
            cth =  ctherm(q, self.telescope.X, T, lam, dlam,
                          diam_collect, self.telescope.Tsys,
                          self.telescope.emissivity)
        else:
            cth = np.zeros_like(cs)

        # Additional background from sky for ground-based observations

        if self.skyflux is not None:
            if self.GROUND == "ESO":
                # Use ESO SKCALC
                wl_sky, Isky = get_sky_flux()
                # Convolve to instrument resolution
                Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)

            elif self.GROUND == "SKYFLUX":
                # Custom ESO SkyCalc option. See sky_flux.py. Must pass in a skyflux object
                wl_sky = self.skyflux.lam
                Isky = self.skyflux.flux

                Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)
            else:
                # Get SMART computed surface intensity due to sky background
                Itherm  = get_thermal_ground_intensity(lam, dlam, convolution_function)

            # Compute Earth thermal photon count rate
            cthe = ctherm_earth(q, self.telescope.X, T, lam, dlam,
                                diam_collect, Itherm)

            # Add earth thermal photon counts to telescope thermal counts
            cth = cth + cthe

        # Calculate background photon count rate
        cback = cz + cez + cth + cD + cR

        # Save count rates as attributes
        self.cp = cp
        self.cs = cs
        self.cback = cback
        self.cz = cz
        self.cez = cez
        self.cth = cth
        self.cD = cD
        self.cR = cR

        # Flip the switch
        self._computed = True

        ########## Calculate SNR-like Quantities ##########

        # Count PLANET photons per eclipse
        Nplan = self.tdur * 1 * cp

        # Count STELLAR photons per eclipse
        Nstar = self.tdur * 1 * cs

        # Count BACKGROUND photons per eclipse
        Nback = self.tdur * 1 * cback

        # Calculate SNR on missing planet photons in one eclipse
        #   This formula assumes a homogeneous planet disk (i.e. no limb darkening / hot-spots),
        #   and comes from standard error propigation on the missing photons due to the
        #   star occulting the planet calculation in terms of observables
        SNR1 = Nplan / np.sqrt( (1+1./self.nout)*Nstar + 1./self.nout*Nplan+(1+1./self.nout)*Nback)

        # Calculate SNR on missing planet photons in ntran eclipses
        SNRn =  np.sqrt(self.ntran) * SNR1

        # Calculate the SECONDS required to observe a given SNR as a function of the spectral res
        tSNR = self.wantsnr**2 * ( (1+1./self.nout)*cs + 1./self.nout*cp+(1+1./self.nout)*cback ) / cp**2

        # Calculate the NUMBER OF ECLIPSES required to observe a given SNR as a function of the spectral res
        nSNR = self.wantsnr**2 * ( (1+1./self.nout) * self.tdur * cs + 1./self.nout * self.tdur * cp + (1+1./self.nout)*self.tdur*cback ) / (self.tdur * cp)**2

        # Save SNR quantities as attributes
        self.SNR1 = SNR1
        self.SNRn = SNRn
        self.tSNR = tSNR
        self.nSNR = nSNR

        # Save additional stuff
        self.lam = lam
        self.dlam = dlam
        self.FpFslr = FpFslr
        self.FpFshr = FpFs

        # Create fake data
        self.make_fake_data()

        return

    def make_fake_data(self):
        """
        Make a fake dataset by sampling from a Gaussian.

        Attributes
        ----------
        SNRn : array
            S/N in ``ntran`` eclipses
        obs : array
            Observed emission specrum with noise
        sig : array
            Observed uncertainties on emission spectrum
        """

        # Ensure that simulation has been run
        assert self._computed

        # Calculate SNR on missing planet photons in ntran eclipses
        self.SNRn =  np.sqrt(self.ntran) * self.SNR1

        # Generate synthetic observations
        self.sig = self.FpFslr / self.SNRn
        self.obs = random_draw(self.FpFslr, self.sig)

    def recalc_wantsnr(self, wantsnr = None):
        """
        Recalculate the time and number of eclipses required to achieve a
        user specified SNR via `wantsnr`.

        Attributes
        ----------
        tSNR : array
            Exposure time to ``wantsnr`` [s]
        nSNR : array
            Number of eclipses to ``wantsnr``
        """

        assert self._computed

        if wantsnr is not None:
            self.wantsnr = wantsnr

        # Calculate the SECONDS required to observe a given SNR as a function of the spectral res
        self.tSNR = self.wantsnr**2 * ( (1+1./self.nout)*self.cs \
                                  + 1./self.nout*self.cp+(1+1./self.nout)*self.cback )\
                                  / self.cp**2

        # Calculate the NUMBER OF ECLIPSES required to observe a given SNR as a function of the spectral res
        self.nSNR = self.wantsnr**2 * ( (1+1./self.nout) * self.tdur * self.cs \
                                  + 1./self.nout * self.tdur * self.cp \
                                  + (1+1./self.nout)*self.tdur*self.cback ) \
                                  / (self.tdur * self.cp)**2

        return


    def plot_spectrum(self, SNR_threshold = 0.0, Nsig = None, ax0 = None,
                      err_kws = {"fmt" : ".", "c" : "k", "alpha" : 1},
                      plot_kws = {"lw" : 1.0, "c" : "C4", "alpha" : 0.5},
                      draw_box = True):
        """
        Plot noised emission spectrum.

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

        m = [self.SNRn > SNR_threshold]

        scale = 1e6

        if ax0 is None:
            # Create Plot
            fig, ax = plt.subplots(figsize = (10,8))
            ax.set_xlabel(r"Wavelength [$\mu$m]")
            ax.set_ylabel(r"Eclipse Depth $(F_p / F_{\star})$ [ppm]")
        else:
            ax = ax0

        #ax.plot(lam, scale*RpRs2, alpha = 1.0, ls = "steps-mid")
        ax.errorbar(self.lam[m], scale*self.obs[m], yerr=scale*self.sig[m], zorder = 100, **err_kws)
        #ax.set_yscale("log")

        # Set ylim
        if Nsig is not None:
            mederr = scale*np.nanmedian(self.sig)
            medy = scale*np.nanmedian(self.obs)
            ax.set_ylim([medy - Nsig*mederr, medy + Nsig*mederr])

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        ax.plot(self.lamhr, scale*self.FpFshr, **plot_kws)

        ax.set_ylim(ylims)
        ax.set_xlim(xlims)


        if draw_box:
            text = "%i eclipses \n %i m \n %i\%% throughput" %(self.ntran, self.telescope.diameter, 100*self.telescope.throughput)
            ax.text(0.02, 0.975, text, transform=ax.transAxes, ha = "left", va = "top",
                    bbox=dict(boxstyle="square", fc="w", ec="k", alpha=0.9), zorder=101)

        #ax.legend()

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_SNRn(self, ax0 = None, plot_kws = {"ls" : "steps-mid"}):
        """
        Plot the S/N on the Eclipse Depth as a function of wavelength.

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

        ax.plot(self.lam, self.SNRn, **plot_kws)
        #ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("S/N on Eclipse Depth")
        #ax.legend()

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_ntran_to_wantsnr(self, ax0 = None,
                              plot_kws = {"ls" : "steps-mid", "alpha" : 1.0}):
        """
        Plot the number of eclipses to get a SNR on the eclipse depth as
        a function of wavelength.

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
            ax.set_ylabel("Eclipses to S/N = %i on Eclipse Depth" %self.wantsnr)
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.nSNR, **plot_kws)

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_time_to_wantsnr(self, ax0 = None, plot_kws = {"ls" : "steps-mid", "alpha" : 1.0}):
        """
        Plot the time to get a SNR on the eclipse depth as
        a function of wavelength.

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
            ax.set_ylabel("Time to S/N = %i on Eclipse Depth [s]" %self.wantsnr)
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.tSNR, **plot_kws)

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_count_rates(self, ax0 = None):
        """
        Plot the photon count rate for all sources.

        Parameters
        ----------
        ax0 : `matplotlib.axes`
            Optional axis to provide

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
            ax.set_ylabel("Photons / s")
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.cp, label = "Planet", ls = "dashed")
        ax.plot(self.lam, self.cs, label = "Star")
        ax.plot(self.lam, self.cback, label = "Total Bkg")
        ax.plot(self.lam, self.cz, label = "SS Zodi")
        ax.plot(self.lam, self.cez, label = "Exo-Zodi")
        ax.plot(self.lam, self.cth, label = "Thermal Bkg")
        ax.plot(self.lam, self.cD, label = "Dark")
        ax.plot(self.lam, self.cR, label = "Read")

        if ax0 is None:
            leg = ax.legend(fontsize = 14, ncol = 2)
            return fig, ax
        else:
            return

class TransitNoise(object):
    """
    Simulate exoplanet transit transmission spectroscopy with a next-generation
    telescope.

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
    tdur : float
        Transit duration [s]
    ntran : float
        Number of transits
    nout : float
        Number of out-of-transit transit durations to observe
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

    """
    def __init__(self, tdur    = 3432.,  # TRAPPIST-1e
                       telescope = Telescope(),
                       planet = Planet(),
                       star = Star(),
                       skyflux = None,
                       ntran   = 1,
                       nout    = 1,
                       wantsnr = 1000.0,
                       NIR     = True,
                       THERMAL = True,
                       GROUND  = False,
                       vod     = False,
                       IMAGE   = False,
                       ZODI = True):
        self.telescope = telescope
        self.planet    = planet
        self.star      = star
        self.skyflux   = skyflux
        self.tdur      = tdur
        self.ntran     = ntran
        self.nout      = nout
        self.wantsnr   = wantsnr
        self.NIR       = NIR
        self.THERMAL   = THERMAL
        self.GROUND    = GROUND
        self.vod       = vod
        self.IMAGE     = IMAGE
        self.ZODI      = ZODI

        self._computed = False

        return

    def run_count_rates(self, lamhr = None, tdhr = None, Fshr = None):
        """
        Calculate the photon count rates and signal to noise on a
        transmission spectrum observation

        Parameters
        ----------

        lamhr : numpy.ndarray
            Wavelength [$\mu$m]
        tdhr : numpy.ndarray
            Transit Depth $(Rp/Rs)^2$
        Fshr : numpy.ndarray
            Flux density incident at the planet's TOA [W/m$^2$/$\mu$]

        Calling ``run_count_rates()`` creates the following attributes for
        the ``TransitNoise`` instance:

        Attributes
        ----------
        lamhr : array
            Wavelength [$\mu$m]
        tdhr : array
            Transit Depth $(Rp/Rs)^2$
        Fshr : array
            Flux density incident at the planet's TOA [W/m$^2$/$\mu$]
        cs : array
            Stellar photon count rate [photons/s]
        cback : array
            Background photon count rate [photons/s]
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
        cmiss : array
            Occulted stellar photon count rate [photons/s]
        SNR1 : array
            S/N for one transit
        SNRn : array
            S/N for ``ntran`` transits
        tSNR : array
            Exposure time to ``wantsnr`` [s]
        nSNR : array
            Number of transits to ``wantsnr``
        lam : array
            Observed wavelength grid [$\mu$m]
        dlam : array
            Observed wavelength grid widths [$\mu$m]
        RpRs2 : array
            Low-res transit depth

        """

        self.lamhr = lamhr
        self.tdhr = tdhr
        self.Fshr = Fshr

        if self.telescope.A_collect is None:
            diam_collect = self.telescope.diameter
        else:
            diam_collect = 2. * (self.telescope.A_collect / np.pi)**0.5

        # Set the convolution function
        convolution_function = downbin_spec

        # Does the telescope object already have a wavelength grid?
        if (self.telescope.lam is None) or (self.telescope.dlam is None):
            # Create wavelength grid
            lam, dlam = construct_lam(self.telescope.lammin,
                                      self.telescope.lammax,
                                      self.telescope.resolution)
        else:
            # Use existing grids
            lam = self.telescope.lam
            dlam = self.telescope.dlam

        # Set Quantum Efficiency
        q = set_quantum_efficiency(lam,
                                   self.telescope.qe,
                                   NIR=self.NIR,
                                   vod=self.vod)

        # Set Dark current and Read noise
        De = set_dark_current(lam,
                              self.telescope.darkcurrent,
                              self.telescope.lammax,
                              self.telescope.Tdet,
                              NIR=self.NIR)
        Re = set_read_noise(lam,
                            self.telescope.readnoise,
                            NIR=self.NIR)

        # Set Angular size of lenslet
        theta = set_lenslet(lam,
                            self.telescope.lammin,
                            diam_collect,
                            self.telescope.X,
                            NIR=self.NIR)

        # Set throughput
        #sep  = r/d*np.sin(alpha*np.pi/180.)*np.pi/180./3600. # separation in radians
        #T = set_throughput(lam, Tput, diam, sep, IWA, OWA, lammin, FIX_OWA=FIX_OWA, SILENT=SILENT)
        T = self.telescope.throughput * np.ones_like(lam)

        # Apply wavelength-dependent throuput, if needed
        if self.telescope.Tput_lam is not None:
            # Bin input throughput curve to native res
            Tlam = np.interp(lam, self.telescope.Tput_lam[0], self.telescope.Tput_lam[1])
            # Multiply into regular throughput
            T = T * Tlam

        # Apply wavelength-dependent quantum efficiency, if needed
        if self.telescope.qe_lam is not None:
            # Bin input QE curve to native res
            qlam = np.interp(lam, self.telescope.qe_lam[0], self.telescope.qe_lam[1])
            # Multiply into regular QE
            q = q * qlam

        # Modify throughput by atmospheric transmission if GROUND-based
        if self.GROUND:
            # Use SMART calc
            #Tatmos = set_atmos_throughput(lam, dlam, convolution_function)
            Tatmos = set_atmos_throughput_skyflux(self.skyflux.lam, self.skyflux.trans, lam, dlam, convolution_function)
            # Multiply telescope throughput by atmospheric throughput
            T = T * Tatmos

        # Degrade and doppler shift transit and stellar spectrum
        tdhr_shifted = doppler_shift(lamhr, tdhr, self.star.vs)
        RpRs2 = convolution_function(tdhr_shifted,lamhr,lam,dlam=dlam)

        Fshr_shifted = doppler_shift(lamhr, Fshr, self.star.vs)

        # Calculate intensity of the star [W/m^2/um/sr]
        if Fshr is None:
            # Using a blackbody
            Bstar = planck(self.star.Teff, lam)
        else:
            # Using provided TOA stellar flux
            Fslr = convolution_function(Fshr_shifted, lamhr, lam, dlam=dlam)
            Bstar = Fslr / ( np.pi*(self.star.Rs*u.Rsun.in_units(u.km)/\
                           (self.planet.a*u.AU.in_units(u.km)))**2. )

        # Solid angle in steradians
        omega_star = np.pi*(self.star.Rs*u.Rsun.in_units(u.km)/\
                           (self.planet.distance*u.pc.in_units(u.km)))**2.
        omega_planet = np.pi*(self.planet.Rp*u.Rearth.in_units(u.km)/\
                             (self.planet.distance*u.pc.in_units(u.km)))**2.

        # Fluxes at earth [W/m^2/um]
        Fs = Bstar * omega_star
        #Fback = jwst_background(lam)
        Fstar_miss = Fs * RpRs2

        # Fraction of planetary signal in Airy pattern
        fpa = 1.0   # No fringe pattern here --> all of stellar psf falls on CCD

        ########## Calculate Photon Count Rates ##########

        # Stellar photon count rate
        cs = cstar(q, fpa, T, lam, dlam, Fs, diam_collect)

        # Missing photon count rate (is this a thing? it is now!)
        cmiss = Fstar_miss*dlam*(lam*1e-6)/(h*c)*T*(np.pi * (0.5*diam_collect)**2)

        if self.ZODI:
            # Solar System Zodi count rate
            cz =  czodi(q, self.telescope.X, T, lam, dlam,
                        diam_collect, self.planet.MzV)

            # Exo-Zodi count rate
            cez =  cezodi(q, self.telescope.X, T, lam, dlam, diam_collect,
                          self.planet.a,
                          Fstar(lam, self.star.Teff, self.star.Rs, 1., AU=True),
                          self.planet.Nez, self.planet.MezV)
        else:
            cz = np.zeros_like(cs)
            cez = np.zeros_like(cs)

        # Dark current count rate
        cD =  cdark(De, self.telescope.X, lam,
                    diam_collect, theta,
                    self.telescope.DNHpix, IMAGE=self.IMAGE)

        # Read noise count rate
        cR =  cread(Re, self.telescope.X, lam, diam_collect,
                    theta, self.telescope.DNHpix, self.telescope.Dtmax,
                    IMAGE=self.IMAGE)

        # Thermal background count rate
        if self.THERMAL:
            # telescope internal thermal count rate
            cth =  ctherm(q, self.telescope.X, T, lam, dlam,
                          diam_collect, self.telescope.Tsys,
                          self.telescope.emissivity)
        else:
            cth = np.zeros_like(cs)

        # Additional background from sky for ground-based observations
        if self.skyflux is not None:

            if self.GROUND == "ESO":
                # Use the standard ESO SKCALC output
                wl_sky, Isky = get_sky_flux()
                # Convolve to instrument resolution
                Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)
            elif self.GROUND == "SKYFLUX":
                # Custom ESO SkyCalc option. See sky_flux.py. Must pass in a skyflux object
                wl_sky = self.skyflux.lam
                Isky = self.skyflux.flux

                Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)


            else:
                # Get SMART computed surface intensity due to sky background
                Itherm  = get_thermal_ground_intensity(lam, dlam, convolution_function)

            # Compute Earth thermal photon count rate
            cthe = ctherm_earth(q, self.telescope.X, T, lam, dlam,
                                diam_collect, Itherm)

            if self.THERMAL:
                # Add earth thermal photon counts to telescope thermal counts
                cth = cth + cthe
            else:
                cth = np.zeros_like(cs)

        # Calculate background photon count rate
        cback = cz + cez + cth + cD + cR

        # Save count rates as attributes
        self.cs = cs
        self.cback = cback
        self.cz = cz
        self.cez = cez
        self.cth = cth
        self.cD = cD
        self.cR = cR
        self.cmiss = cmiss
        self.Tatmos = Tatmos

        # Flip the switch
        self._computed = True

        ########## Calculate SNR-like Quantities ##########

        # Count STELLAR photons per transit
        Nstar = self.tdur * 1 * cs

        # Count BACKGROUND photons per transit
        Nback = self.tdur * 1 * cback

        # Calculate SNR on missing stellar photons in one transit
        #   This formula assumes a homogeneous stellar disk (i.e. no limb darkening),
        #   and comes from standard error propigation on the missing photons due to the
        #   planet occulting the star calculation in terms of observables
        SNR1 = (Nstar * RpRs2) / np.sqrt((1 + 1./self.nout - RpRs2) * Nstar + (1 + 1./self.nout) * Nback)

        # Calculate SNR on missing stellar photons in ntran transits
        SNRn =  np.sqrt(self.ntran) * SNR1

        # Calculate the SECONDS required to observe a given SNR as a function of the spectral res
        tSNR = self.wantsnr**2 * ((1 + 1./self.nout - RpRs2) * cs + (1 + 1./self.nout) * cback) / (cs * RpRs2)**2

        # Calculate the NUMBER OF TRANSITS required to observe a given SNR as a function of the spectral res
        nSNR = self.wantsnr**2 * ((1 + 1./self.nout - RpRs2) *  self.tdur * cs + (1 + 1./self.nout) * self.tdur * cback) / (self.tdur * cs * RpRs2)**2

        # Save SNR quantities as attributes
        self.SNR1 = SNR1
        self.SNRn = SNRn
        self.tSNR = tSNR
        self.nSNR = nSNR

        # Save additional stuff
        self.lam = lam
        self.dlam = dlam
        self.RpRs2 = RpRs2
        self.tdhr = convolution_function(tdhr, lamhr, lam, dlam=dlam)

        # Create fake data
        self.make_fake_data()

        return

    def make_fake_data(self):
        """
        Make a fake dataset by sampling from a Gaussian.

        Attributes
        ----------
        SNRn : array
            S/N in ``ntran`` transits
        obs : array
            Observed transit depth with noise
        sig : array
            Observed uncertainties on transit depth
        """

        # Ensure that simulation has been run
        assert self._computed

        # Calculate SNR on missing stellar photons in ntran transits
        self.SNRn =  np.sqrt(self.ntran) * self.SNR1
        self.sig = self.RpRs2 / self.SNRn

        # Generate synthetic observations

        self.obs = random_draw(self.RpRs2, self.sig)

    def recalc_wantsnr(self, wantsnr = None):
        """
        Recalculate the time and number of transits required to achieve a
        user specified SNR via `wantsnr`.

        Attributes
        ----------
        tSNR : array
            Exposure time to ``wantsnr`` [s]
        nSNR : array
            Number of transits to ``wantsnr``
        """

        assert self._computed

        if wantsnr is not None:
            self.wantsnr = wantsnr

        # Calculate the SECONDS required to observe a given SNR as a function of the spectral res
        self.tSNR = self.wantsnr**2 * ((1 + 1./self.nout - self.RpRs2) \
                                       * self.cs + (1 + 1./self.nout) \
                                       * self.cback) / (self.cs * self.RpRs2)**2

        # Calculate the NUMBER OF TRANSITS required to observe a given SNR as a function of the spectral res
        self.nSNR = self.wantsnr**2 * ((1 + 1./self.nout - self.RpRs2) \
                                       *  self.tdur * self.cs + (1 + 1./self.nout) \
                                       * self.tdur * self.cback) / (self.tdur * self.cs * self.RpRs2)**2

        return


    def plot_spectrum(self, SNR_threshold = 1.0, Nsig = 6.0, ax0 = None,
                      err_kws = {"fmt" : ".", "c" : "k", "alpha" : 1},
                      plot_kws = {"lw" : 1.0, "c" : "C4", "alpha" : 0.5},
                      draw_box = True):
        """
        Plot noised transmission spectrum.

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

        m = [self.SNRn > SNR_threshold]

        scale = 1e6

        if ax0 is None:
            # Create Plot
            fig, ax = plt.subplots(figsize = (10,8))
            ax.set_xlabel(r"Wavelength [$\mu$m]")
            ax.set_ylabel("Transit Depth $(R_p / R_{\star})^2$ [ppm]")
        else:
            ax = ax0

        #ax.plot(lam, scale*RpRs2, alpha = 1.0, ls = "steps-mid")
        ax.errorbar(self.lam[m], scale*self.obs[m], yerr=scale*self.sig[m], zorder = 100, **err_kws)
        #ax.set_yscale("log")

        # Set ylim
        mederr = scale*np.nanmedian(self.sig)
        medy = scale*np.nanmedian(self.obs)
        ax.set_ylim([medy - Nsig*mederr, medy + Nsig*mederr])

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        ax.plot(self.lamhr, scale*self.tdhr, **plot_kws)

        ax.set_ylim(ylims)
        ax.set_xlim(xlims)


        if draw_box:
            text = "%i transits \n %i m \n %i\%% throughput" %(self.ntran, self.telescope.diameter, 100*self.telescope.throughput)
            ax.text(0.02, 0.975, text, transform=ax.transAxes, ha = "left", va = "top",
                    bbox=dict(boxstyle="square", fc="w", ec="k", alpha=0.9), zorder=101)

        #ax.legend()

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_SNRn(self, ax0 = None, plot_kws = {"ls" : "steps-mid"}):
        """
        Plot the S/N on the Transit Depth as a function of wavelength.

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

        ax.plot(self.lam, self.SNRn, **plot_kws)
        #ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("S/N on Transit Depth")
        #ax.legend()

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_ntran_to_wantsnr(self, ax0 = None,
                              plot_kws = {"ls" : "steps-mid", "alpha" : 1.0}):
        """
        Plot the number of transits to get a SNR on the transit depth as
        a function of wavelength.

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
            ax.set_ylabel("Transits to S/N = %i on Transit Depth" %self.wantsnr)
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.nSNR, **plot_kws)

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_time_to_wantsnr(self, ax0 = None, plot_kws = {"ls" : "steps-mid", "alpha" : 1.0}):
        """
        Plot the time to get a SNR on the transit depth as
        a function of wavelength.

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
            ax.set_ylabel("Time to S/N = %i on Transit Depth [s]" %self.wantsnr)
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.tSNR, **plot_kws)

        if ax0 is None:
            return fig, ax
        else:
            return

    def plot_count_rates(self, ax0 = None):
        """
        Plot the photon count rate for all sources.

        Parameters
        ----------
        ax0 : `matplotlib.axes`
            Optional axis to provide

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
            ax.set_ylabel("Photons / s")
            ax.set_yscale("log")
        else:
            ax = ax0

        ax.plot(self.lam, self.cmiss, label = "Occulted", ls = "dashed")
        ax.plot(self.lam, self.cs, label = "Star")
        ax.plot(self.lam, self.cback, label = "Total Bkg")
        ax.plot(self.lam, self.cz, label = "SS Zodi")
        ax.plot(self.lam, self.cez, label = "Exo-Zodi")
        ax.plot(self.lam, self.cth, label = "Thermal Bkg")
        ax.plot(self.lam, self.cD, label = "Dark")
        ax.plot(self.lam, self.cR, label = "Read")

        if ax0 is None:
            leg = ax.legend(fontsize = 14, ncol = 2)
            return fig, ax
        else:
            return

def get_earth_trans_spectrum():
    '''
    Get the transmission spectrum of the Earth around the Sun.

    Returns
    -------
    lam : `numpy.ndarray`
        Wavelength grid [um]
    tdepth : `numpy.ndarray`
        Transit depth (Rp/Rs)^2
    fplan : `numpy.ndarray`
        TOA planet flux [W/m^2/um]
    fstar : `numpy.ndarray`
        Stellar flux at planet [W/m^2/um]
    '''

    # Read in transit data
    here = os.path.join(os.path.dirname(__file__))
    plus = "planets/earth_avg_hitran2012_300_100000cm.trnst"
    data = np.loadtxt(os.path.join(here, plus))

    # Parse
    lam = data[:,0]
    tdepth = data[:,3]

    # Read in flux data
    plus = "planets/earth_avg_hitran2012_300_100000cm_toa.rad"
    data = np.loadtxt(os.path.join(here, plus))

    # Parse
    fplan = data[:,3]
    fstar = data[:,2]

    return lam, tdepth, fplan, fstar
