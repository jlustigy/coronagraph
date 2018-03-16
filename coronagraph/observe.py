from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import os

from .count_rates_wrapper import count_rates_wrapper
from .teleplanstar import Telescope, Planet, Star
from .plot_setup import setup

__all__ = ['generate_observation', 'planetzoo_observation',
           'process_noise', 'exptime_band', 'interp_cont_over_band',
           'plot_interactive_band', 'random_draw', 'get_earth_reflect_spectrum']

planetdir = "planets/"
relpath = os.path.join(os.path.dirname(__file__), planetdir)

def get_earth_reflect_spectrum():
    """
    Get the geometric albedo spectrum of the Earth around the Sun.
    This was produced by Tyler Robinson using the VPL Earth Model
    (Robinson et al., 2011)

    Returns
    -------
    lamhr : numpy.ndarray
    Ahr : numpy.ndarray
    fstar : numpy.ndarray
    """

    fn = 'earth_quadrature_radiance_refl.dat'
    model = np.loadtxt(os.path.join(relpath,fn), skiprows=8)
    lamhr = model[:,0]
    radhr = model[:,1]
    fstar = model[:,2]
    reflhr = model[:,3]
    Ahr   = np.pi*(np.pi*radhr/fstar) # hi-resolution reflectivity

    return lamhr, Ahr, fstar

def planetzoo_observation(name='earth', telescope=Telescope(), planet=Planet(), itime=10.0,
                            planetdir = relpath, plot=True, savedata=False, saveplot=False,
                            ref_lam=0.55, THERMAL=False):
    """Uses coronagraph model to observe Solar System planets

    Parameters
    ----------
    name : str (optional)
        Name of the planet (e.g. "venus", "earth", "archean", "mars",
        "earlymars", "hazyarchean", "earlyvenus", "jupiter", "saturn", "uranus",
        "neptune")
    telescope : Telescope (optional)
        Telescope object to be used for observation
    planet : Planet (optional)
        Planet object to be used for observation
    itime : float (optional)
        Integration time (hours)
    planetdir : str
        Location of planets/ directory
    plot : bool (optional)
        Make plot flag
    savedata : bool (optional)
        Save output as data file
    saveplot : bool (optional)
        Save plot as PDF
    ref_lam : float (optional)
        Wavelength at which SNR is computed

    Returns
    -------
    lam : array
        Observed wavelength array (microns)
    spec : array
        Observed reflectivity spectrum
    sig : array
        Observed 1-sigma error bars on spectrum
    """

    import os
    try:
        l = os.listdir(planetdir)
    except OSError:
        print("Error in planetzoo_observation(): planetdir does not exist in current location. \nSet planetdir='location/of/planets/'")
        return None

    whichplanet = name
    startype = planet.star
    tag = name

    if True:

        #if whichplanet == 'earth':
        fn = 'earth_quadrature_radiance_refl.dat'
        model = np.loadtxt(os.path.join(planetdir,fn), skiprows=8)
        lamhr = model[:,0]
        radhr = model[:,1]
        solhr = model[:,2]
        reflhr = model[:,3]
        Ahr   = np.pi*(np.pi*radhr/solhr) # hi-resolution reflectivity
        planet.Rp    = 1.0     # Earth radii
        planet.r     = 1.0     # semi-major axis (AU)
        lamhr0 = lamhr
        solhr0 = solhr

        if whichplanet == 'venus':
            fn = 'Venus_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 0.95     #Earth radii
            planet.r     = 0.72     #semi-major axis (AU)

        if whichplanet == 'archean':
            fn = 'ArcheanEarth_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 1.0     #Earth radii
            planet.r     = 1.0     #semi-major axis (AU)

        if whichplanet == 'earlymars':
            fn = 'EarlyMars_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 0.53     #Earth radii
            planet.r     = 1.52     #semi-major axis (AU)

        if whichplanet == 'hazyarchean':
            fn = 'Hazy_ArcheanEarth_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 1.0     #Earth radii
            planet.r     = 1.0     #semi-major axis (AU)

        if whichplanet == 'earlyvenus':
            fn = 'EarlyVenus_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 0.95     #Earth radii
            planet.r     = 0.72     #semi-major axis (AU)

        if whichplanet == 'jupiter':
            fn = planetdir+'Jupiter_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 10.86     #Earth radii
            planet.r     = 5.20     #semi-major axis (AU)

        if whichplanet == 'saturn':
            fn = planetdir+'Saturn_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 9.00     #Earth radii
            planet.r     = 9.54     #semi-major axis (AU)

        if whichplanet == 'uranus':
            fn = 'Uranus_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 3.97     #Earth radii
            planet.r     = 19.19     #semi-major axis (AU)

        if whichplanet == 'neptune':
            fn = 'Neptune_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 3.85     #Earth radii
            planet.r     = 30.07     #semi-major axis (AU)


        if whichplanet == 'mars':
            fn = 'Mars_geo_albedo.txt'
            model = np.loadtxt(os.path.join(planetdir,fn))
            lamhr = model[:,0]
            Ahr   = model[:,1]
            planet.Rp    = 0.53     #Earth radii
            planet.r     = 1.52     #semi-major axis (AU)

        if whichplanet == 'fstarozone':
            print('fstarozone functionality not yet added')
            #fn = 'fstarcloudy.sav'
            #fn = 'F2V_5.e-1fCO2_1.e6H2Volc_1.e10BIF.out_toa.rad'
            #READCOL, 'planets/'+fn, lamhr, wnhr, sunhr, fluxhr
            #restore, 'planets/'+fn
            #lamhr = reverse(wl_)
            #ahr = reverse(cloudyalb_)
            #Ahr = (2/3.) * fluxhr/(sunhr/2.) #geometric albedo & phase corrections
            #Rp    = 1.0     #Earth radii
            #r     = 1.72    #semi-major axis (AU)



    # star parameters
    if startype == '':
        Teff  = 5780.   #Sun-like Teff (K)
        Rs    = 1.      #star radius in solar radii
    if  startype == 'sun':
        Teff  = 5780.   #Sun-like Teff (K)
        Rs    = 1.      #star radius in solar radii
    if  startype == 'f2v':
        Teff  = 7050.   #Sun-like Teff (K)
        Rs    = 1.3      #star radius in solar radii

    star = Star(Teff=Teff, Rs=Rs)

    # interpolate stellar flux to planet wavelength grid
    solhr = np.interp(lamhr, lamhr0, solhr0)

    mask = (Ahr > 0.0) & np.isfinite(Ahr)

    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR \
        = count_rates_wrapper(Ahr[mask], lamhr[mask], solhr[mask], telescope, planet, star,
                              COMPUTE_LAM=True, THERMAL=THERMAL, otype=2)

    # Calculate background photon count rate
    cb = (cz + cez + csp + cD + cR + cth)

    # Calculate the SNR of observation
    time = itime * 3600. # Convert hours to seconds
    #SNR = calc_SNR(time, cp, cb)

    # Generate noisy spectrum by drawing data points from a normal distribution
    #spec, sig = draw_noisy_spec(A, SNR)

    # Calculate SNR, sigma, and noised-up spectrum
    spec, sig, SNR = process_noise(time, Cratio, cp, cb)

    if plot:
        fig, ax = plot_coronagraph_spectrum(lam, spec, sig, itime, planet.distance, ref_lam, SNR, truth=Cratio)

    # Save Synthetic data file (wavelength, albedo, error) if savedata=True
    if savedata:
        data_tag = 'observed_'+tag+'.txt'
        y_sav = np.array([lam,spec,sig])
        np.savetxt(data_tag, y_sav.T)
        print('Saved: '+data_tag)

    # Return Synthetic data and high-res spec
    return lam, spec, sig

def generate_observation(wlhr, Ahr, solhr, itime, telescope, planet, star,
                         ref_lam=0.55, tag='', plot=True, saveplot=False, savedata=False,
                         THERMAL=False, wantsnr=10):
    """
    Generic wrapper function for `count_rates`.

    Parameters
    ----------
    wlhr : float
        Wavelength array (microns)
    Ahr : float
        Geometric albedo spectrum array
    itime : float
        Integration time (hours)
    telescope : Telescope
        Telescope object
    planet : Planet
        Planet object
    star : Star
        Star object
    tag : string
        ID for output files
    plot : boolean
        Set to True to make plot
    saveplot : boolean
        Set to True to save the plot as a PDF
    savedata : boolean
        Set to True to save data file of observation

    Returns
    -------
    lam : array
        Wavelength grid for observed spectrum
    dlam: array
        Wavelength grid widths for observed spectrum
    A : array
        Low res albedo spectrum
    spec : array
        Observed albedo spectrum
    sig : array
        One sigma errorbars on albedo spectrum
    SNR : array
        SNR in each spectral element


    Note
    ----
    If `saveplot=True` then plot will be saved
    If `savedata=True` then data will be saved
    """

    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR \
        = count_rates_wrapper(Ahr, wlhr, solhr, telescope, planet, star,
                              COMPUTE_LAM=True, THERMAL=THERMAL, otype=2,
                              wantsnr=wantsnr)

    # Calculate background photon count rate
    cb = (cz + cez + csp + cD + cR + cth)

    # Calculate the SNR of observation
    time = itime * 3600. # Convert hours to seconds

    # Calculate SNR, sigma, and noised-up spectrum
    spec, sig, SNR = process_noise(time, Cratio, cp, cb)

    #SNR = calc_SNR(time, cp, cb)

    # Generate noisy spectrum by drawing data points from a normal distribution
    #spec, sig = draw_noisy_spec(A, SNR)


    if plot:
        fig, ax = plot_coronagraph_spectrum(lam, spec, sig, itime, planet.distance, ref_lam, SNR, truth=Cratio)

    # Save Synthetic data file (wavelength, albedo, error) if savedata=True
    if savedata:
        data_tag = 'observed_'+tag+'.txt'
        y_sav = np.array([lam,spec,sig])
        np.savetxt(data_tag, y_sav.T)
        print('Saved: '+data_tag)

    # Return Synthetic data and high-res spec

    return lam, dlam, Cratio, spec, sig, SNR

'''
def smart_observation(radfile, itime, telescope, planet, star,
                         ref_lam=0.55, tag='', plot=True, saveplot=False, savedata=False,
                         THERMAL=False, wantsnr=10.):
    """Uses coronagraph noise model to create an observation of high resolution SMART output.

    Parameters
    ----------
    radfile : string
        Location and name of file to be read in
    itime : float
        Integration time (hours)
    telescope : Telescope
        Telescope object
    planet : Planet
        Planet object
    star : Star
        Star object
    tag : string
        ID for output files
    plot : boolean
        Set to True to make plot
    saveplot : boolean
        Set to True to save the plot as a PDF
    savedata : boolean
        Set to True to save data file of observation

    Returns
    ----------
    lam : array
        Wavelength grid of observed spectrum
    spec : array
        Albedo grid of observed spectrum
    sig : array
        One sigma errorbars on albedo spectrum
    rwl : array
        Wavelength grid of SMART output
    Ahr : array
        Albedo grid of SMART output

    Output
    ---------
    If saveplot=True then plot will be saved
    If savedata=True then data will be saved
    """

    # try importing readsmart
    try:
        import readsmart as rs
    except ImportError:
        print("Module 'readsmart' not found. Please install on your local machine \
        to proceed with this function. The source can be found at: \
        https://github.com/jlustigy/readsmart")
        return None, None, None, None, None

    # Read-in .rad file
    wlhr, wno, solar_spec, TOA_flux, rad_streams = rs.rad(radfile,getdata=True)

    # Calculate Hi-res reflectivity spectrum
    Ahr = (TOA_flux / solar_spec) #* np.pi / planet.Phi

    # Calculate photon count rates
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR \
        = count_rates_wrapper(Ahr, wlhr, solhr, telescope, planet, star,
                              COMPUTE_LAM=True, THERMAL=THERMAL, otype=2,
                              wantsnr=wantsnr)

    # Calculate background photon count rate
    cb = (cz + cez + csp + cD + cR + cth)

    # Calculate the SNR of observation
    time = itime * 3600. # Convert hours to seconds
    #SNR = calc_SNR(time, cp, cb)

    # Generate noisy spectrum by drawing data points from a normal distribution
    #spec, sig = draw_noisy_spec(A, SNR)

    # Calculate SNR, sigma, and noised-up spectrum
    spec, sig, SNR = process_noise(time, Cratio, cp, cb)

    if plot:
        fig, ax = plot_coronagraph_spectrum(lam, spec, sig, itime, planet.distance, ref_lam, SNR, truth=Cratio)

    # Save Synthetic data file (wavelength, albedo, error) if savedata=True
    if savedata:
        data_tag = 'observed_smart_'+tag+'.txt'
        y_sav = np.array([lam,spec,sig])
        np.savetxt(data_tag, y_sav.T)
        print('Saved: '+data_tag)

    # Return Synthetic data and high-res spec

    return lam, spec, sig, wlhr, Ahr
'''

def plot_coronagraph_spectrum(wl, ofrat, sig, itime, d, ref_lam, SNR,
                              truth=None,
                              xlim=None, ylim=None,
                              title="",
                              save=False, tag=""):
    """
    Plot synthetic data from the coronagraph model

    Parameters
    ----------
    wl : array
    ofrat : array
    sig : array
    itime : array
    d : array
    ref_lam : array
    SNR : array
    truth : array (optional)
    xlim : list (optional)
    ylim : list (optional)
    title : str (optional)
    save : bool (optional)
    tag : str (optional)

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axis

    Note
    ----
    Only returns ```fig, ax`` if ``save = False``
    """

    # Create figure
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    # Set string for plot text
    if itime > 2.0:
        timestr = "{:.0f}".format(itime)+' hours'
    else:
        timestr = "{:.0f}".format(itime*60)+' mins'
    plot_text = r'Distance = '+"{:.1f}".format(d)+' pc'+\
    '\n Integration time = '+timestr

    # If a reference wavelength is specified then return the SNR at that wl
    # corresponding to the integration time given
    if ref_lam:
        ireflam = find_nearest(wl,ref_lam)
        ref_SNR = SNR[ireflam]
        plot_text = plot_text + '\n SNR = '+"{:.1f}".format(ref_SNR)+\
            ' at '+"{:.2f}".format(wl[ireflam])+r' $\mu$m'

    if truth is not None:
        ax.plot(wl, truth*1e9, lw=2.0, color="purple", alpha=0.7, ls="steps-mid")
    ax.errorbar(wl, ofrat*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)

    ax.set_ylabel(r"F$_p$/F$_s$ ($\times 10^9$)")
    ax.set_xlabel("Wavelength [$\mu$m]")
    ax.set_title(title)
    ax.text(0.99, 0.99, plot_text,\
         verticalalignment='top', horizontalalignment='right',\
         transform=ax.transAxes,\
         color='black', fontsize=20)

    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_xlim(xlim)

    if save:
        fig.savefig(title+tag+".pdf")
        return
    else:
        return fig, ax

def process_noise(Dt, Cratio, cp, cb):
    """
    Computes SNR, noised data, and error on noised data.

    Parameters
    ----------
    Dt : float
        Telescope integration time in seconds
    Cratio : array
        Planet/Star flux ratio in each spectral bin
    cp : array
        Planet Photon count rate in each spectral bin
    cb : array
        Background Photon count rate in each spectral bin

    Returns
    -------
    cont : array
        Noised Planet/Star flux ratio in each spectral bin
    sigma : array
        One-sigma errors on flux ratio in each spectral bin
    SNR : array
        Signal-to-noise ratio in each spectral bin
    """

    # Calculate signal-to-noise assuming background subtraction (the "2")
    SNR  = cp*Dt/np.sqrt((cp + 2*cb)*Dt)

    # Calculate 1-sigma errors
    sigma= Cratio/SNR

    # Add gaussian noise to flux ratio
    cont = Cratio + np.random.randn(len(Cratio))*sigma

    return cont, sigma, SNR

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

def random_draw(val, sig):
    """
    Draw fake data points from model ``val`` with errors ``sig``
    """
    if type(val) is np.ndarray:
        return val + np.random.randn(len(val))*sig
    elif (type(val) is float) or (type(val) is int) or (type(val) is np.float64):
        return val + np.random.randn(1)[0]*sig

def interp_cont_over_band(lam, cp, icont, iband):
    """
    Interpolate the continuum of a spectrum over a masked absorption or emission
    band.

    Parameters
    ----------
    lam : array
        Wavelength grid (abscissa)
    cp : array
        Planet photon count rates or any spectrum
    icont : list
        Indicies of continuum (neighboring points)
    iband : list
        Indicies of spectral feature (the band)

    Returns
    -------
    ccont : list
        Continuum planet photon count rates across spectral feature, where
        len(ccont) == len(iband)
    """
    # Linearly interpolate continuum points to band
    ccont = np.interp(lam[sorted(iband)], lam[sorted(icont)], cp[sorted(icont)])
    return ccont

def exptime_band(cp, ccont, cb, iband, SNR=5.0):
    """
    Calc the exposure time necessary to get a given S/N on a molecular band
    following Eqn 7 from Robinson et al. 2016.

    Parameters
    ----------
    cp :
        Planet count rate
    ccont :
        Continuum count rate
    cb :
        Background count rate
    iband :
        Indicies of molecular band
    SNR :
        Desired signal-to-noise ratio on molecular band

    Returns
    -------
    texp : float
        Telescope exposure time [hrs]
    """

    numerator = np.sum(cp[iband] + 2.*cb[iband])
    denominator = np.power(np.sum(np.fabs(ccont - cp[iband])),2)

    return np.power(SNR, 2) * numerator / denominator / 3600.0

def SNR_band(cp, ccont, cb, iband, itime=10.):
    """
    Calc the exposure time necessary to get a given S/N on a molecular band
    following Eqn 7 from Robinson et al. 2016.

    Parameters
    ----------
    cp :
        Planet count rate
    ccont :
        Continuum count rate
    cb :
        Background count rate
    iband :
        Indicies of molecular band
    itime :
        Integration time [hours]

    Returns
    -------
    snr : float
        SNR to detect band given exposure time
    """

    denominator = np.power(np.sum(cp[iband] + 2.*cb[iband]), 0.5)
    numerator = np.sum(np.fabs(ccont - cp[iband]))

    return np.power(itime*3600., 0.5) * numerator / denominator

click = 0
icont = []
iband = []
def plot_interactive_band(lam, Cratio, cp, cb, itime=None, SNR=5.0):
    """
    Makes an interactive spectrum plot for the user to identify all observed
    spectral points that make up a molecular band. Once the plot is active,
    press 'c' then select neighboring points in the Continuum, press 'b' then
    select all points in the Band, then press 'd' to perform the calculation.

    Parameters
    ----------
    lam : array
        Wavelength grid
    Cratio : array
        Planet-to-star flux contrast ratio
    cp : array
        Planetary photon count rate
    cb : array
        Background photon count rate
    itime : float (optional)
        Fiducial exposure time for which to calculate the SNR
    SNR : float (optional)
        Fiducial SNR for which to calculate the exposure time
    """

    # Turn off interactive plotting shortcut keys
    plt.rcParams['keymap.back'] = ''

    verbose = False
    OGC = "black"
    CC = "orange"
    BC = "green"

    def onpick(event):
        """
        Funtion to handle picked points in interactive plot window
        """
        ind = event.ind

        global icont
        global iband

        # Picked continuum value
        if click == 1:
            # Check if picked value is in either list
            if ind[0] in icont:
                # remove from continuum list
                icont.remove(ind[0])
                # reset point to original color
                ax.scatter(lam[ind], Cratio[ind]*1e9, s=40.0, color=OGC, alpha=1.0, zorder=100)
            elif ind[0] in iband:
                # remove from band list
                iband.remove(ind[0])
                # add to continuum list
                icont.append(ind[0])
                # plot point in new color
                ax.scatter(lam[ind], Cratio[ind]*1e9, s=40.0, color=CC, alpha=1.0, zorder=100)
            else:
                # add to continuum list
                icont.append(ind[0])
                # plot point in new color
                ax.scatter(lam[ind], Cratio[ind]*1e9, s=40.0, color=CC, alpha=1.0, zorder=100)
            if verbose: print("icont:", icont)
            ax.figure.canvas.draw()

        # Picked band value
        elif click == 2:
            # Check if picked value is in either list
            if ind[0] in iband:
                # remove from band list
                iband.remove(ind[0])
                # reset point to original color
                ax.scatter(lam[ind], Cratio[ind]*1e9, s=40.0, color=OGC, alpha=1.0, zorder=100)
            elif ind[0] in icont:
                # remove from continuum list
                icont.remove(ind[0])
                # add to band list
                iband.append(ind[0])
                # plot point in new color
                ax.scatter(lam[ind], Cratio[ind]*1e9, s=40.0, color=BC, alpha=1.0, zorder=100)
            else:
                # add to band list
                iband.append(ind[0])
                # plot point in new color
                ax.scatter(lam[ind], Cratio[ind]*1e9, s=40.0, color=BC, alpha=1.0, zorder=100)
            if verbose: print("iband:", iband)
            ax.figure.canvas.draw()

    def on_key(event):
        """
        Function to handle key presses in interactive plot window
        """
        global click
        global iband
        global icont
        print('you pressed', event.key)
        if event.key == "c":
            click = 1
        if event.key == "b":
            click = 2
        if event.key == "d":
            click = 0
            if (len(icont) < 2) or (len(iband) < 1):
                print("Must select at least two continuum points and one band point.")
            elif (min(iband) < min(icont)) or (max(iband) > max(icont)):
                print("Must select at least one point on either side of the band of interest")
            else:
                # interpolate continuum planet counts to band wavelengths
                ccont = interp_cont_over_band(lam, cp, icont, iband)
                ccrat = interp_cont_over_band(lam, Cratio, icont, iband)
                #"""
                # plot new interpolated points
                ax.scatter(lam[iband], ccrat*1e9, s=40.0, color=BC, alpha=1.0, zorder=100)
                ax.figure.canvas.draw()
                #"""
                # Calculate the exposure time and SNR
                if SNR is not None:
                    etime = exptime_band(cp, ccont, cb, iband, SNR=SNR)
                    print("Exposure Time = %.5f hours to get SNR = %.5f" %(etime, SNR))
                if itime is not None:
                    eSNR = SNR_band(cp, ccont, cb, iband, itime=itime)
                    print("SNR = %.5f in a %.5f hour exposure" %(eSNR, itime))

    # Create figure
    fig, ax = plt.subplots(figsize=(12,6))

    # Set string for plot text
    """
    if itime > 2.0:
        timestr = "{:.0f}".format(itime)+' hours'
    else:
        timestr = "{:.0f}".format(itime*60)+' mins'
    """
    #plot_text = r'Distance = '+"{:.1f}".format(planet.distance)+' pc'+\
    #'\n Integration time = '+timestr

    #if ref_lam:
    #    ireflam = find_nearest(wl,ref_lam)
    #    ref_SNR = SNR[ireflam]
    #    plot_text = plot_text + '\n SNR = '+"{:.1f}".format(ref_SNR)+\
    #        ' at '+"{:.2f}".format(wl[ireflam])+r' $\mu$m'

    ax.plot(lam, Cratio*1e9, "-", lw=2.0, color=OGC, alpha=0.7, ls="steps-mid")
    ax.scatter(lam, Cratio*1e9, s=40.0, color=OGC, alpha=1.0, picker=True)
    #ax.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)

    ax.set_ylabel(r"F$_p$/F$_s$ ($\times 10^9$)")
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    #ax.set_title("")
    #ax.text(0.99, 0.99, plot_text,\
    #     verticalalignment='top', horizontalalignment='right',\
    #     transform=ax.transAxes,\
    #     color='black', fontsize=20)

        #fig, ax = plt.subplots()
        #col = ax.scatter(x, y, 100*s, c, picker=True)
        ##fig.savefig('pscoll.eps')

    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
