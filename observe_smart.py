# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:20:26 2015

@author: jlustigy
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rcParams['font.size'] = 20.0
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=30, usetex=True)

# Import local modules
from read_smart import read_rad
from call_atlast_noise import call_atlast_noise

def generate_observation(wlhr, spechr, etime, tag='', startype='sun', \
    lammin=0.3,lammax=2.0, Res=70., Tput=0.2, diam=8.0, Tsys=274.,\
    IWA=0.5, OWA=30000., d=10., Nez=1., emis=0.9,\
    C=1.e-10, alpha=90., Phi=1./np.pi, Rp=1.0, r=1.0,\
    ref_lam=False, saveplot=False, savedata=False, scale=3.*np.pi/2.):
        
    '''
    Parameters
    ----------
    infile : string
        Location and name of file to be read in
    etime : float
        Integration time (hours)
    tag : string
        ID for output files
    rad : boolean
        Set to True if infile is a SMART output file (*.rad)
    atm : boolean
        Set to True if infile is an atmospheric structure file (*.atm)
        Note: Currently unavailable (8/15)
    startype : string
        Type of star to use for noise model (options: 'sun', 'f2v')
    lammin : float
        Minimum wavelength (microns)
    lammax : float
        Maximum wavelength (microns)
    Res : float
        Resolving power of telescope
    Tput : float 
        Telescope throughput 
    diam : float
        Telescope diameter (m)
    Tsys : float
        Telescope temperature (K)
    IWA : float
        Telescope inner working angle (lambda/D)
    OWA : float
        Telescope outer working angle (lambda/D)
    d : float
        Distance to planet (pc)
    Nez : float
        Number of exo-zodis (SS zodi)
    emis : float
        Telescope/system emissivity
    C : float
        Raw Contrast
    alpha : float
        Phase angle (default 90, quadrature)
    Phi : float
        Phase function (default 1/pi, quadrature)
    Rp : float
        Planet radius (default 1.0, Earth)
    r : float
        Planet semi-major axis (default 1.0, Earth)
    ref_lam : boolean/float
        Wavelength (microns) at which SNR is printed in plot
    saveplot : boolean
        Set to True to save the plot as a PDF
    scale : float
        Scaling factor for the albedo 
        For now, 3*pi/2 converts to apparant albedo
    
    Returns
    ----------
    lam : float64
        Wavelength grid of observed spectrum
    spec : float65
        Albedo grid of observed spectrum
    sig : float64
        One sigma errorbars on albedo spectrum
    rwl : float64
        Wavelength grid of SMART output
    Ahr : float64
        Albedo grid of SMART output
    
    Output
    ---------
    If saveplot=True then plot will be saved
    If savedata=True then data will be saved
    '''

    # Compose High-Res Albedo spectrum
    Ahr = spechr * scale

    # Call ATLAST Noise model
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
    call_atlast_noise(Ahr=Ahr, lamhr=wlhr, startype=startype, \
        lammin=lammin,lammax=lammax, Res=Res, Tput=Tput, diam=diam, Tsys=Tsys,\
        IWA=IWA, OWA=OWA, d=d, Nez=Nez, emis=emis, wantsnr=10.,\
        C=C, alpha=alpha, Phi=Phi, Rp=Rp, r=r)    

    # Calculate background photon count rate
    cb = (cz + cez + csp + cD + cR + cth)

    # Calculate the SNR of observation
    time = etime * 3600. # Convert hours to seconds
    SNR = calc_SNR(time, cp, cb)   
    
    # Generate noisy spectrum by drawing data points from a normal distribution
    spec, sig = draw_noisy_spec(A, SNR)

    # Set string for plot text
    plot_text = r'Distance = '+"{:.1f}".format(d)+' pc'+\
    '\n Integration time = '+"{:.0f}".format(etime)+' hours'

    # If a reference wavelength is specified then return the SNR at that wl 
    # corresponding to the integration time given
    if ref_lam:
        ref_SNR = SNR[find_nearest(lam,ref_lam)] 
        plot_text = plot_text + '\n SNR = '+"{:.1f}".format(ref_SNR)+\
            ' at '+"{:.1f}".format(ref_lam)+r' $\mu$m'    
    
    # Plot observed spectrum; save pdf if saveplot=True 
    plot_tag = 'plots/smart_data_'+tag+'.pdf'
    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.plot(lam, A, alpha=0.7, color='orange', drawstyle='steps-mid', lw=2.0)
    ax0.errorbar(lam, spec, yerr=sig, fmt='o', color='k')
    ax0.set_ylabel('Radiance [W m$^{2}$ $\mu$m$^{-1}$ sr$^{-1}$]')
    ax0.set_xlabel('Wavelength [$\mu$m]')
    ax0.set_xlim([lammin,lammax])
    #ax0.set_ylim([-0.01,1.01])
    ax0.text(0.01, 0.99, plot_text,\
         verticalalignment='top', horizontalalignment='left',\
         transform=ax0.transAxes,\
         color='black', fontsize=20)
    if saveplot:
        fig.savefig(plot_tag)
        print 'Saved: '+plot_tag
    
    # Save Synthetic data file (wavelength, albedo, error) if savedata=True
    if savedata:
        data_tag = 'output/smart_data_'+tag+'.txt'
        y_sav = np.array([lam,spec,sig])
        np.savetxt(data_tag, y_sav.T)
        print 'Saved: '+data_tag
    
    # Return Synthetic data and high-res spec
    
    return lam, spec, sig, wlhr, Ahr
    
def calc_SNR(itime, cp, cb, poisson=2.):
    
    cnoise = cp + poisson*cb
    SNR = cp*itime / np.sqrt(cnoise*itime)
    
    return SNR

def draw_noisy_spec(spectrum, SNR, apparent=False):
    
    if apparent:
        # Scale geometric albedo to apparent albedo (as if lambertian sphere) for quadrature
        scale = 3.*np.pi/2.
    else:
        scale = 1.0  
    
    # Compute "sigma"
    sigma = scale * spectrum / SNR
    # Draw data points from normal distribution    
    spec_noise = np.random.randn(len(spectrum))*sigma + scale*spectrum
    
    return spec_noise, sigma

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

