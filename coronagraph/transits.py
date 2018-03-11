#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate exoplanet transmission spectroscopy with a blank mask coronagraph.
This uses the same telesope and detector as the coronagraph, but does not
block the star's light. As a result, stellar photons dominate the noise
budget.

"""

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from .noise_routines import *
from .degrade_spec import *
from .observe import random_draw
from .plot_setup import setup
setup()

__all__ = ["TransitNoise"]

h = 6.62607004e-34
c = 2.998e8

class TransitNoise(object):
    """
    Attributes
    ----------
    tdur : float
        Transit duration [s]
    r : float
        Semi-major axis [AU]
    d : float
        Distance to system [pc]
    Rp : float
        Planet radius [Earth radii]
    Rs : float
        Stellar radius [Solar radii]
    Teff : float
        Stellar effective temperature [K]
    ntran : float
        Number of transits
    nout : float
        Number of out-of-transit transit durations to observe
    lammin : float
        Minimum wavelength [$\mu$m]
    lammax : float
        Maximum wavelength [$\mu$m]
    Nez : float
        Number of exo-zodi [Solar System Zodi]
    diam : float
        Telescope diameter [m]
    Tput : float
        Telescope throughput
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

    """
    def __init__(self, tdur    = 3432.,  # TRAPPIST-1e
                       r       = 0.0281,
                       d       = 12.2,
                       Rp      = 0.918,
                       Rs      = 1.0,
                       Teff    = 2500.,
                       ntran   = 1,
                       nout    = 1,
                       lammin  = 0.5,
                       lammax  = 2.0,
                       Res     = 70.0,
                       Nez     = 1.0,
                       diam    = 15.,
                       Tput    = 0.2,
                       Tsys    = 260.0,
                       Tdet    = 50.0,
                       emis    = 0.9,
                       De      = 1e-4,
                       DNHpix  = 3.0,
                       Re      = 0.1,
                       Dtmax   = 1.0,
                       X       = 1.5,
                       qe      = 0.9,
                       MzV     = 23.0,
                       MezV    = 22.0,
                       wantsnr = 1000.0,
                       NIR     = True,
                       THERMAL = True,
                       GROUND  = False,
                       vod     = False,
                       IMAGE   = False):

        self.tdur    = tdur
        self.ntran   = ntran
        self.nout    = nout
        self.lammin  = lammin
        self.lammax  = lammax
        self.Res     = Res
        self.r       = r
        self.d       = d
        self.Nez     = Nez
        self.Rp      = Rp
        self.Teff    = Teff
        self.Rs      = Rs
        self.diam    = diam
        self.Tput    = Tput
        self.Tsys    = Tsys
        self.Tdet    = Tdet
        self.emis    = emis
        self.De      = De
        self.DNHpix  = DNHpix
        self.Re      = Re
        self.Dtmax   = Dtmax
        self.X       = X
        self.qe      = qe
        self.MzV     = MzV
        self.MezV    = MezV
        self.wantsnr = wantsnr
        self.NIR     = NIR
        self.THERMAL = THERMAL
        self.GROUND  = GROUND
        self.vod     = vod
        self.IMAGE   = IMAGE

        self._computed = False

        return

    def run_count_rates(self, lamhr, tdhr, Fshr):
        """
        Calculate the photon count rates and signal to noise on an observation

        Parameters
        ----------
        lamhr : numpy.ndarray
            Wavelength [$\mu$m]
        tdhr : numpy.ndarray
            Transit Depth $(Rp/Rs)^2$
        Fshr : numpy.ndarray
            Flux density incident at the planet's TOA [W/m$^2$/$\mu$]
        """

        self.lamhr = lamhr
        self.tdhr = tdhr
        self.Fshr = Fshr

        # Set the convolution function
        convolution_function = downbin_spec

        # Create wavelength grid
        lam, dlam = construct_lam(self.lammin, self.lammax, self.Res)

        # Set Quantum Efficiency
        q = set_quantum_efficiency(lam, self.qe, NIR=self.NIR, vod=self.vod)

        # Set Dark current and Read noise
        De = set_dark_current(lam, self.De, self.lammax, self.Tdet, NIR=self.NIR)
        Re = set_read_noise(lam, self.Re, NIR=self.NIR)

        # Set Angular size of lenslet
        theta = set_lenslet(lam, self.lammin, self.diam, self.X, NIR=self.NIR)

        # Set throughput
        #sep  = r/d*np.sin(alpha*np.pi/180.)*np.pi/180./3600. # separation in radians
        #T = set_throughput(lam, Tput, diam, sep, IWA, OWA, lammin, FIX_OWA=FIX_OWA, SILENT=SILENT)
        T = self.Tput * np.ones_like(lam)

        # Modify throughput by atmospheric transmission if GROUND-based
        if self.GROUND:
            # Use SMART calc
            Tatmos = set_atmos_throughput(lam, dlam, convolution_function)
            # Multiply telescope throughput by atmospheric throughput
            T = T * Tatmos

        # Degrade transit and stellar spectrum
        RpRs2 = convolution_function(tdhr,lamhr,lam,dlam=dlam)

        # Calculate intensity of the star [W/m^2/um/sr]
        if Fshr is None:
            # Using a blackbody
            Bstar = planck(self.Tstar, lam)
        else:
            # Using provided TOA stellar flux
            Fslr = convolution_function(Fshr, lamhr, lam, dlam=dlam)
            Bstar = Fslr / ( np.pi*(self.Rs*u.Rsun.in_units(u.km)/(self.r*u.AU.in_units(u.km)))**2. )

        # Solid angle in steradians
        omega_star = np.pi*(self.Rs*u.Rsun.in_units(u.km)/(self.d*u.pc.in_units(u.km)))**2.
        omega_planet = np.pi*(self.Rp*u.Rearth.in_units(u.km)/(self.d*u.pc.in_units(u.km)))**2.

        # Fluxes at earth [W/m^2/um]
        Fs = Bstar * omega_star
        #Fback = jwst_background(lam)
        Fstar_miss = Fs * RpRs2

        # Fraction of planetary signal in Airy pattern
        fpa = 1.0   # No fringe pattern here --> all of stellar psf falls on CCD

        ########## Calculate Photon Count Rates ##########

        # Stellar photon count rate
        cs = cstar(q, fpa, T, lam, dlam, Fs, self.diam)

        # Missing photon count rate (is this a thing? it is now!)
        cmiss = Fstar_miss*dlam*(lam*1e-6)/(h*c)*T*(np.pi * (0.5*self.diam)**2)

        # Solar System Zodi count rate
        cz =  czodi(q, self.X, T, lam, dlam, self.diam, self.MzV)

        # Exo-Zodi count rate
        cez =  cezodi(q, self.X, T, lam, dlam, self.diam, self.r, \
            Fstar(lam, self.Teff, self.Rs, 1., AU=True), self.Nez, self.MezV)

        # Dark current count rate
        cD =  cdark(De, self.X, lam, self.diam, theta, self.DNHpix, IMAGE=self.IMAGE)

        # Read noise count rate
        cR =  cread(Re, self.X, lam, self.diam, theta, self.DNHpix, self.Dtmax, IMAGE=self.IMAGE)

        # Thermal background count rate
        if self.THERMAL:
            cth =  ctherm(q, self.X, lam, dlam, self.diam, self.Tsys, self.emis)                      # internal thermal count rate
        else:
            cth = np.zeros_like(cs)

        # Additional background from sky for ground-based observations
        if self.GROUND:

            if self.GROUND == "ESO":
                # Use ESO SKCALC
                wl_sky, Isky = get_sky_flux()
                # Convolve to instrument resolution
                Itherm = convolution_function(Isky, wl_sky, lam, dlam=dlam)
            else:
                # Get SMART computed surface intensity due to sky background
                Itherm  = get_thermal_ground_intensity(lam, dlam, convolution_function)

            # Compute Earth thermal photon count rate
            cthe = ctherm_earth(q, self.X, lam, dlam, self.diam, self.Itherm)

            # Add earth thermal photon counts to telescope thermal counts
            cth = cth + cthe

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

        # Create fake data
        self.make_fake_data()

        return

    def make_fake_data(self):
        """
        Make a fake dataset by sampling from a Gaussian.
        """

        # Ensure that simulation has been run
        assert self._computed

        # Calculate SNR on missing stellar photons in ntran transits
        self.SNRn =  np.sqrt(self.ntran) * self.SNR1

        # Generate synthetic observations
        self.sig = self.RpRs2 / self.SNRn
        self.obs = random_draw(self.RpRs2, self.sig)

    def recalc_wantsnr(self, wantsnr = None):
        """
        Recalculate the time and number of transits required to achieve a
        user specified SNR via `wantsnr`.
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
        mederr = scale*np.median(self.sig)
        medy = scale*np.median(self.obs)
        ax.set_ylim([medy - Nsig*mederr, medy + Nsig*mederr])

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        ax.plot(self.lamhr, scale*self.tdhr, **plot_kws)

        ax.set_ylim(ylims)
        ax.set_xlim(xlims)


        if draw_box:
            text = "%i transits \n %i m \n %i\%% throughput" %(self.ntran, self.diam, 100*self.Tput)
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

    def plot_ntran_to_wantsnr(self, ax0 = None, plot_kws = {"ls" : "steps-mid", "alpha" : 1.0}):
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
