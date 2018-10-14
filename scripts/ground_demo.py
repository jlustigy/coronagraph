#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
ground_demo.py |github|
-----------------------

A demo simulation of Earth at 10 pc using a 30-m ground-based telescope setup.

  .. role:: raw-html(raw)
     :format: html

  .. |github| replace:: :raw-html:`<a href = "https://github.com/jlustigy/coronagraph/blob/master/scripts/ground_demo.py"><i class="fa fa-github" aria-hidden="true"></i></a>`

'''
from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

# Import some standard python packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pdb
import sys, os

# The location to *this* file
RELPATH = os.path.dirname(__file__)

# Import coronagraph model
import coronagraph as cg
from coronagraph import plot_setup
plot_setup.setup()

def _test():
    '''

    '''
    run()

def run():
    '''
    Run an example :class:`coronagraph` spectrum assuming a ground-based
    observatory.

    Example
    -------
    >>> import ground_demo
    >>> ground_demo.run()

      .. plot::
        :align: center

        from scripts import ground_demo
        from coronagraph import plot_setup
        plot_setup.setup()
        ground_demo._test()

    '''

    ################################
    # PARAMETERS
    ################################

    # Integration time (hours)
    Dt = 500.

    # Planet params
    alpha = 90.     # phase angle at quadrature
    Phi   = cg.teleplanstar.lambertPhaseFunction(alpha)      # phase function at quadrature (already included in SMART run)
    Rp    = 1.0     # Earth radii
    r     = 1.0     # semi-major axis (AU)

    # Stellar params
    Teff  = 5780   # Sun-like Teff (K)
    Rs    = 1.     # Star radius in solar radii

    # Planetary system params
    d    = 10.0    # Distance to system (pc)
    Nez  = 1.      # Number of exo-zodis

    # Telescope parameters
    lammin = 0.3
    lammax = 2.5
    Res    = 70.0
    diam   = 30.0
    Tput   = 0.05
    C      = 1e-10
    IWA    = 1.0
    OWA    = 40.0
    Tsys   = 269.0
    Tdet   = 50.0
    emis   = 0.9
    De     = 1e-4
    DNHpix = 3.0
    Re     = 0.1
    Dtmax  = 1.0
    X      = 1.5
    qe     = 0.9
    MzV    = 23.0
    MezV   = 22.0


    # Plot params
    plot = True
    ref_lam = 0.55
    title = ""
    ylim = [-0.1, 0.3]
    xlim =  None
    tag = "GroundIR_500hr_new"

    # Save params
    savefile = False
    saveplot = False


    ################################
    # READ-IN DATA
    ################################

    # Read-in spectrum file
    fn = os.path.join(RELPATH, '../coronagraph/planets/earth_quadrature_radiance_refl.dat')
    model = np.loadtxt(fn, skiprows=8)
    lamhr = model[:,0]
    radhr = model[:,1]
    solhr = model[:,2]

    # Calculate hi-resolution reflectivity
    Ahr   = np.pi*(np.pi*radhr/solhr)

    ################################
    # RUN CORONAGRAPH MODEL
    ################################

    # Run coronagraph with default LUVOIR telescope (aka no keyword arguments)
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        cg.count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
                       GROUND = True,
                       THERMAL = True,
                       lammin = lammin,
                       lammax = lammax,
                       Res    = Res   ,
                       diam   = diam  ,
                       Tput   = Tput  ,
                       C      = C     ,
                       IWA    = IWA   ,
                       OWA    = OWA   ,
                       Tsys   = Tsys  ,
                       Tdet   = Tdet  ,
                       emis   = emis  ,
                       De     = De    ,
                       DNHpix = DNHpix,
                       Re     = Re    ,
                       Dtmax  = Dtmax ,
                       X      = X     ,
                       qe     = qe    ,
                       MzV    = MzV   ,
                       MezV   = MezV  )


    # Calculate background photon count rates
    cb = (cz + cez + csp + cD + cR + cth)

    # Convert hours to seconds
    Dts = Dt * 3600.

    # Calculate signal-to-noise assuming background subtraction (the "2")
    SNR  = cp*Dts/np.sqrt((cp + 2*cb)*Dts)

    # Calculate 1-sigma errors
    sig= Cratio/SNR

    # Add gaussian noise to flux ratio
    spec = Cratio + np.random.randn(len(Cratio))*sig

    ################################
    # PLOTTING
    ################################

    if plot:

        plot_setup.setup()

        # Create figure
        fig, ax = plt.subplots(figsize = (10,8))

        # Set string for plot text
        if Dt > 2.0:
            timestr = "{:.0f}".format(Dt)+' hours'
        else:
            timestr = "{:.0f}".format(Dt*60)+' mins'
        plot_text = r'Distance = '+"{:.1f}".format(d)+' pc'+\
        '\n Integration time = '+timestr

        # If a reference wavelength is specified then return the SNR at that wl
        # corresponding to the integration time given
        if ref_lam:
            ireflam = (np.abs(lam - ref_lam)).argmin()
            ref_SNR = SNR[ireflam]
            plot_text = plot_text + '\n SNR = '+"{:.1f}".format(ref_SNR)+\
                ' at '+"{:.2f}".format(lam[ireflam])+r' $\mu$m'

        # Draw plot
        ax.plot(lam, Cratio*1e9, lw=2.0, color="purple", alpha=0.7, ls="steps-mid")
        ax.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)

        # Set labels
        ax.set_ylabel(r"F$_p$/F$_s$ ($\times 10^9$)", fontsize = 25)
        ax.set_xlabel("Wavelength [$\mu$m]", fontsize = 25)
        ax.set_title(title)
        ax.text(0.98, 0.98, plot_text,\
             verticalalignment='top', horizontalalignment='right',\
             transform=ax.transAxes,\
             color='black', fontsize=20,
             bbox=dict(boxstyle="square", fc="w", ec="k", alpha=0.9), zorder=101)

        # Adjust x,y limits
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)

        # Save plot if requested
        if saveplot:
            plot_tag = "ground_demo_"+title+tag+".png"
            fig.savefig(plot_tag)
            print('Saved: ' + plot_tag)
        else:
            plt.show()

    ################################
    # SAVING
    ################################

    # Save Synthetic data file (wavelength, albedo, error) if requested
    if savefile:
        data_tag = 'ground_demo_'+tag+'.txt'
        y_sav = np.array([lam,spec,sig])
        np.savetxt(data_tag, y_sav.T)
        print('Saved: ' + data_tag)

if __name__ == '__main__':

    run()
