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

    # Calculate background photon count rate
    cb = (cz + cez + csp + cD + cR + cth)

    # Calculate SNR, sigma, and noised-up spectrum
    time = 3600.
    spec, sig, SNR = cg.process_noise(time, Cratio, cp, cb)


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


    lam, dlam, Cratio, spec, sig, SNR = cg.observe.generate_observation(
                             lamhr, Ahr, solhr, 10.0, cg.Telescope(),
                             cg.Planet(), cg.Star(),
                             ref_lam=0.55, tag='', plot=False, saveplot=False,
                             savedata=False, THERMAL=True, wantsnr=10)
    return
