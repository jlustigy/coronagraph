"""
"""

import coronagraph as cg
import numpy as np

def test_defult_teleplanstar():
    """
    Parameters
    ----------

    Returns
    -------
    """

    telescope = cg.Telescope()
    telescope = cg.Telescope.default_luvoir()
    telescope = cg.Telescope.default_habex()
    telescope = cg.Telescope.default_wfirst()
    print(telescope)

    planet = cg.Planet()
    print(planet)

    star = cg.Star()
    print(star)

    return

def test_count_rates():
    """
    Parameters
    ----------

    Returns
    -------
    """

    # Planet params
    alpha = 90.     # phase angle at quadrature
    Phi   = cg.noise_routines.lambertPhaseFunction(alpha)  # phase function
    Rp    = 1.0     # Earth radii
    r     = 1.0     # semi-major axis (AU)

    # Stellar params
    Teff  = 5780.   # Sun-like Teff (K)
    Rs    = 1.      # star radius in solar radii

    # Planetary system params
    d    = 10.     # distance to system (pc)
    Nez  = 1.      # number of exo-zodis

    # Create hi-res wavelength grid
    lamhr = np.linspace(0.2, 2.5, 1e4)

    # Create fake hi-resolution reflectivity
    Ahr   = 0.1 + 0.1 * lamhr * np.sin(np.pi * lamhr / 5.)

    # Create fake stellar flux
    Bstar = cg.noise_routines.planck(Teff, lamhr)
    solhr = Bstar * np.pi * (6.957e5/1.496e8)**2.

    ################################
    # RUN CORONAGRAPH MODEL
    ################################

    # Run coronagraph with default LUVOIR telescope (aka no keyword arguments)
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        cg.count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
                       lammax=1.6)

    # Run coronagraph with all kwargs turned on
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        cg.count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
                       lammax=1.6, THERMAL = True, GROUND = True, vod=True)

    # Run wrapper
    out = cg.count_rates_wrapper(Ahr, lamhr, solhr, cg.Telescope(), cg.Planet(),
                                 cg.Star())

    # Calculate background photon count rate
    cb = (cz + cez + csp + cD + cR + cth)

    # Calculate SNR, sigma, and noised-up spectrum
    time = 3600.
    spec, sig, SNR = cg.process_noise(time, Cratio, cp, cb)

    # Test calc_snr
    snr = cg.observe.calc_SNR(time, cp, cb, poisson=2.)

    # Test draw_noisy_spec
    spec_noise, sigma = cg.observe.draw_noisy_spec(Cratio, snr, apparent=False)

    assert np.all(np.isfinite(spec))
    assert np.all(np.isfinite(sig))
    assert np.all(np.isfinite(SNR))

    return

def test_planetzoo():
    """
    Parameters
    ----------

    Returns
    -------
    """

    lam, spec, sig = cg.observe.planetzoo_observation(name='earth', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='venus', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='mars', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='archean', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='earlymars', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='hazyarchean', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='earlyvenus', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='jupiter', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='saturn', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='uranus', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='neptune', plot = False)
    lam, spec, sig = cg.observe.planetzoo_observation(name='warmuranus', plot = False)

    return

def test_observe():
    """
    Parameters
    ----------

    Returns
    -------
    """

    # Create hi-res wavelength grid
    lamhr = np.linspace(0.2, 2.5, 1e4)

    # Create fake hi-resolution reflectivity
    Ahr   = 0.1 + 0.1 * lamhr * np.sin(np.pi * lamhr / 5.)

    # Create fake stellar flux
    Bstar = cg.noise_routines.planck(5700., lamhr)
    solhr = Bstar * np.pi * (6.957e5/1.496e8)**2.

    # Call generate_observation
    lam, dlam, Cratio, spec, sig, SNR = cg.observe.generate_observation(
                             lamhr, Ahr, solhr, 10.0, cg.Telescope(),
                             cg.Planet(), cg.Star(),
                             ref_lam=0.55, tag='', plot=True, saveplot=False,
                             savedata=False, THERMAL=True, wantsnr=10)

    return

def test_convolution_functions():
    """
    Parameters
    ----------

    Returns
    -------
    """
    # Create hi-res wavelength grid
    lamhr = np.linspace(0.2, 2.5, 1e4)

    # Create fake hi-resolution reflectivity
    Ahr   = 0.1 + 0.1 * lamhr * np.sin(np.pi * lamhr / 5.)

    lam, dlam = cg.noise_routines.construct_lam(0.3, 1.0, dlam = 0.05)

    A1 = cg.degrade_spec(Ahr,lamhr,lam,dlam=dlam)

    A2 = cg.downbin_spec(Ahr,lamhr,lam,dlam=dlam)

    filters = cg.imager.johnson_cousins()
    A3 = cg.convolve_spec(Ahr, lamhr, filters)
    A4 = cg.convolve_spec(Ahr, lamhr, filters, forceTopHat=True)

    return

def test_imager():
    """
    Parameters
    ----------

    Returns
    -------
    """

    # Read all filters
    jc = cg.imager.johnson_cousins()
    jc2 = cg.imager.johnson_cousins2()
    ls = cg.imager.landsat()

    ax = ls.plot()

    # Planet params
    alpha = 90.     # phase angle at quadrature
    Phi   = cg.noise_routines.lambertPhaseFunction(alpha)  # phase function
    Rp    = 1.0     # Earth radii
    r     = 1.0     # semi-major axis (AU)

    # Stellar params
    Teff  = 5780.   # Sun-like Teff (K)
    Rs    = 1.      # star radius in solar radii

    # Planetary system params
    d    = 10.     # distance to system (pc)
    Nez  = 1.      # number of exo-zodis

    # Create hi-res wavelength grid
    lamhr = np.linspace(0.2, 2.5, 1e4)

    # Create fake hi-resolution reflectivity
    Ahr   = 0.1 + 0.1 * lamhr * np.sin(np.pi * lamhr / 5.)

    # Create fake stellar flux
    Bstar = cg.noise_routines.planck(Teff, lamhr)
    solhr = Bstar * np.pi * (6.957e5/1.496e8)**2.

    ################################
    # RUN CORONAGRAPH MODEL
    ################################

    # Run coronagraph with default LUVOIR telescope (aka no keyword arguments)
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        cg.count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
                       lammax=1.6, filter_wheel = ls, mode = "Imaging")

    return

def test_transits():
    """
    Parameters
    ----------

    Returns
    -------
    """

    Rp = 6.371e3
    Rs = 6.957e5
    Teff = 5700

    # Load data
    lamhr, tdhr, fplan, Fshr = cg.get_earth_trans_spectrum()

    # Define parameters
    telescope = cg.Telescope()
    planet = cg.Planet()
    star = cg.Star(Teff = Teff)

    ###############
    ### TRANSIT ###
    ###############

    # Instantiate transit noise model
    tn = cg.TransitNoise(tdur = 8.0 * 60 * 60,
                         ntran = 10.0,
                         nout = 2,
                         telescope = telescope,
                         planet = planet,
                         star = star)

    # Calculate count rates
    tn.run_count_rates(lamhr, tdhr, Fshr)

    tn.ntran = 100
    tn.make_fake_data()
    fig, ax = tn.plot_spectrum()

    fig, ax = tn.plot_SNRn()

    fig, ax = tn.plot_ntran_to_wantsnr()

    # This is the SNR we want on the max difference in planet radius
    wantvsnr = 3

    # Calculate the SNR we want for the transit depths to get the right
    #   SNR on the radius difference
    wantsnr = wantvsnr * np.mean(tn.RpRs2) / (np.max(tn.RpRs2) - np.min(tn.RpRs2))

    tn.recalc_wantsnr(wantsnr = wantsnr)

    fig, ax = tn.plot_ntran_to_wantsnr()

    fig, ax = tn.plot_count_rates()

    ###############
    ### ECLIPSE ###
    ###############

    en = cg.EclipseNoise(tdur = 8.0 * 60 * 60,
                         telescope = telescope,
                         planet = planet,
                         star = star,
                         ntran = 100.,
                         nout = 2.)

    en.run_count_rates(lamhr, fplan, Fshr)

    fig, ax = en.plot_spectrum(SNR_threshold=0.0, Nsig=None)
    fig, ax = en.plot_SNRn()
    fig, ax = en.plot_ntran_to_wantsnr()
    fig, ax = en.plot_count_rates()
    en.recalc_wantsnr(wantsnr = 20)

    return

def test_coronagraphnoise():
    """
    Parameters
    ----------

    Returns
    -------
    """
    noise = cg.CoronagraphNoise()
    lamhr, Ahr, fstar = cg.get_earth_reflect_spectrum()
    noise.run_count_rates(Ahr, lamhr, fstar)
    fig, ax = noise.plot_spectrum()
    fig, ax = noise.plot_SNR()
    fig, ax = noise.plot_time_to_wantsnr()
    return

def test_extras():
    """
    Parameters
    ----------

    Returns
    -------
    """
    cg.plot_setup.setup()
    return
