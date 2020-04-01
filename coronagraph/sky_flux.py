'''
The following script creates an object for the flux of the Earth's sky.

this script requires the user to have the ESO SkyCalc CLI installed via
pip install --user skycalc_cli
https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html
'''

import subprocess
import os
import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np



class SkyFlux:
    '''
    Object containing inputs for ESO SkyCalc calculations

    Note the following moon coordinate constraints:

    |z – zmoon| ≤ ρ ≤ |z + zmoon|

    where ρ=moon/target separation, z=90°−alt, zmoon=90°−moon_alt; z and alt refer to the target

    Parameters
    ----------
    place : string
        output directory
    tag : string
        any identifier you want; will be appended to the end of output files
        can be None
    lam : ndarray
        wavelength (um)
    flux : ndarray
        flux of sky (W/m^2/um)
    trans : ndarray
        transmission spectrum of sky
    airmass : float
        Airmass
        Allowed values: range [1.0, 3.0]
    pwv_mode : string
        "pwv" or "season"
    season : int
        season you are observing
        range [0, 6] (0=all year, 1=dec/jan, 2=feb/mar...)
    time : int
        time of night you are observing
        range [0, 3] (0=all night, 1,2,3 = third of night)
    pwv : float
        Precipitable Water Vapor [mm]
        [0.5,1.0,1.5,2.5,3.5,5.0,7.5,10.0,20.0]
    msolflux : float
        Monthly Averaged Solar Flux [sfu=0.01 MJy]
        range [0.0, +∞] (Monthly averaged solar radio flux density F10.7cm)
    incl_moon : string
        Flag for inclusion of scattered moonlight
        "Y" or "N"
    moon_sun_sep : float
        Separation in deg of Sun and Moon as seen from Earth [degrees]
        range [0.0, 360.0]
    moon_target_sep : float
        Moon-Target Separation in deg [degrees]
        range [0.0, 180.0]
    moon_alt : float
        Moon Altitude over Horizon in deg [degrees]
        range [-90.0, 90.0]
    moon_earth_dist : float
        Moon-Earth Distance, mean=1
        range [0.91, 1.08]
    incl_starlight : string
        Flag for inclusion of scattered starlight
        ["Y", "N"]
    incl_zodiacal : string
        Flag for inclusion of zodiacal light)
        ["Y", "N"]
    ecl_lon : float
        Heliocentric ecliptic longitude [degrees]
        range [-180.0, 180.0]
    ecl_lat
        Ecliptic latitude [degrees]
        range [-90.0, 90.0]
    incl_loweratm : string
        Flag for inclusion of Molecular Emission of Lower Atmosphere
        ["Y", "N"]
    incl_upperatm : string
        Flag for inclusion of Molecular Emission of Upper Atmosphere
        ["Y", "N"]
    incl_airglow : string
        Flag for inclusion of upper airglow continuum
        ["Y", "N"]
    incl_therm : string
        Flag for inclusion of instrumental thermal radiation
        ["Y", "N"]
    therm_t1 : float
        temperature 1 [k]
        range [0.0, +∞]
    therm_e1 : float
        Emmisivity 1
        range [0.0, 1.0]
    therm_t2 : float
        temperature 2 [k]
        range [0.0, +∞]
    therm_e2 : float
        Emmisivity 2
        range [0.0, 1.0]
    therm_t3 : float
        temperature 3 [k]
        range [0.0, +∞]
    therm_e3 : float
        Emmisivity 3
        range [0.0, 1.0]
    vacair : string
        Calculation in vacuum or air?
        ["vac", "air"]
    wmin : float
        Minimum wavelength [nm]
        range [300.0, 30000.0]
    wmax : float
        Maximum wavelength [nm]
        range [300.0, 30000.0]
    wgrid_mode : string
        specify grid mode from below values
        ['fixed_spectral_resolution','fixed_wavelength_step', 'user']
    wdelta : float
        Wavelength sampling dlambda [nm]
        range [0, 30000.0]
    wgrid_user : array
        Array of user-defined wavelength sampling
        [0, 30000.0]
    wres : float
        spectral resolution
        range [0, 1.0e6] (lambda / dlambda)
    lsf_type : string
        Line Spread Function Type
         ["none", "Gaussian", "Boxcar"]
    lsf_gauss_fwhm : float
        wavelength bins
        range [0.0, +∞]
    lsf_boxcar_fwhm : float
        wavelength bins
        range [0.0, +∞]
    observatory : string
        name of observatory
         ["paranal", "lasilla", "3060m"]
    ra : float
        Right Ascension Equatorial Coordinates (degrees)
        [0., 360.]
    dec : float
        Declination Equatorial Coordinates (degrees)
        [-90.0, 90.0]
    date : string | float
        observation date
        YYYY-MM-DDThh:mm:ss | MJD
    '''

    def __init__(self,
                place = './',
                tag = None,
                lam = None,
                flux = None,
                trans = None,
                airmass	=	1.0,
                pwv_mode	=	'pwv',
                season	=	0,
                time	=	0,
                pwv	=	3.5,
                msolflux	=	130.0,
                incl_moon	=	'Y',
                moon_sun_sep	=	90.0,
                moon_target_sep	=	45.0,
                moon_alt	=	45.0,
                moon_earth_dist	=	1.0,
                incl_starlight	=	'Y',
                incl_zodiacal	=	'Y',
                ecl_lon	=	135.0,
                ecl_lat	=	90.0,
                incl_loweratm	=	'Y',
                incl_upperatm	=	'Y',
                incl_airglow	=	'Y',
                incl_therm	=	'N',
                therm_t1	=	0.0,
                therm_e1	=	0.0,
                therm_t2	=	0.0,
                therm_e2	=	0.0,
                therm_t3	=	0.0,
                therm_e3	=	0.0,
                vacair	=	'vac',
                wmin	=	300.0,
                wmax	=	2000.0,
                wgrid_mode	=	'fixed_wavelength_step',
                wdelta	=	0.1,
                wgrid_user = None,
                wres	=	20000,
                lsf_type	=	'none',
                lsf_gauss_fwhm	=	5.0,
                lsf_boxcar_fwhm	=	5.0,
                observatory	=	'paranal',
                ra	=	121.75,
                dec	=	-29.7,
                date	=	'2012-07-17T21:12:14'):

        self.place = place
        self.tag = tag
        self.lam = lam
        self.flux = flux
        self.trans = trans
        self.airmass = airmass
        self.pwv_mode = pwv_mode
        self.season = season
        self.time = time
        self.pwv = pwv
        self.msolflux = msolflux
        self.incl_moon = incl_moon
        self.moon_sun_sep = moon_sun_sep
        self.moon_target_sep = moon_target_sep
        self.moon_alt = moon_alt
        self.moon_earth_dist = moon_earth_dist
        self.incl_starlight = incl_starlight
        self.incl_zodiacal = incl_zodiacal
        self.ecl_lon = ecl_lon
        self.ecl_lat = ecl_lat
        self.incl_loweratm = incl_loweratm
        self.incl_upperatm = incl_upperatm
        self.incl_airglow = incl_airglow
        self.incl_therm = incl_therm
        self.therm_t1 = therm_t1
        self.therm_e1 = therm_e1
        self.therm_t2 = therm_t2
        self.therm_e2 = therm_e2
        self.therm_t3 = therm_t3
        self.therm_e3 = therm_e3
        self.vacair = vacair
        self.wmin = wmin
        self.wmax = wmax
        self.wgrid_mode = wgrid_mode
        self.wdelta = wdelta
        self.wgrid_user = wgrid_user
        self.wres = wres
        self.lsf_type = lsf_type
        self.lsf_gauss_fwhm = lsf_gauss_fwhm
        self.lsf_boxcar_fwhm = lsf_boxcar_fwhm
        self.observatory = observatory
        self.ra = ra
        self.dec = dec
        self.date = date

    def skycalc_params_to_dict(self):
        '''
        creates dicts for skycalc params
        '''
        skycalc_params = {}

        skycalc_params['airmass'] = self.airmass
        skycalc_params['pwv_mode'] = self.pwv_mode
        skycalc_params['season'] = self.season
        skycalc_params['time'] = self.time
        skycalc_params['pwv'] = self.pwv
        skycalc_params['msolflux'] = self.msolflux
        skycalc_params['incl_moon'] = self.incl_moon
        skycalc_params['moon_sun_sep'] = self.moon_sun_sep
        skycalc_params['moon_target_sep'] = self.moon_target_sep
        skycalc_params['moon_alt'] = self.moon_alt
        skycalc_params['moon_earth_dist'] = self.moon_earth_dist
        skycalc_params['incl_starlight'] = self.incl_starlight
        skycalc_params['incl_zodiacal'] = self.incl_zodiacal
        skycalc_params['ecl_lon'] = self.ecl_lon
        skycalc_params['ecl_lat'] = self.ecl_lat
        skycalc_params['incl_loweratm'] = self.incl_loweratm
        skycalc_params['incl_upperatm'] = self.incl_upperatm
        skycalc_params['incl_airglow'] = self.incl_airglow
        skycalc_params['incl_therm'] = self.incl_therm
        skycalc_params['therm_t1'] = self.therm_t1
        skycalc_params['therm_e1'] = self.therm_e1
        skycalc_params['therm_t2'] = self.therm_t2
        skycalc_params['therm_e2'] = self.therm_e2
        skycalc_params['therm_t3'] = self.therm_t3
        skycalc_params['therm_e3'] = self.therm_e3
        skycalc_params['vacair'] = self.vacair
        skycalc_params['wmin'] = self.wmin
        skycalc_params['wmax'] = self.wmax
        skycalc_params['wgrid_mode'] = self.wgrid_mode
        skycalc_params['wgrid_user'] = self.wgrid_user
        skycalc_params['wdelta'] = self.wdelta
        skycalc_params['wres'] = self.wres
        skycalc_params['lsf_type'] = self.lsf_type
        skycalc_params['lsf_gauss_fwhm'] = self.lsf_gauss_fwhm
        skycalc_params['lsf_boxcar_fwhm'] = self.lsf_boxcar_fwhm
        skycalc_params['observatory'] = self.observatory

        return skycalc_params


    def almanac_params_to_dict(self):
        '''
        creates dicts for almanac params
        '''

        almanac_params = {}

        almanac_params['ra'] = self.ra
        almanac_params['dec'] = self.dec
        almanac_params['date'] = self.date
        almanac_params['observatory'] = self.observatory
        return almanac_params


    def write_skycalc_params(self):
        '''
        writes two txt files containing the information that the
        ESO SkyCalc CLI needs to get sky flux
        '''
        try:
            # make the directory
            os.mkdir(self.place)
        except OSError:
            # if the directory is already there, pass
            pass

        # get the name and location  of a file to save to
        if self.tag:
            skycalc_save_fl = self.place + '/' + 'skycalc_params_' + self.tag + '.txt'
            almanac_save_fl = self.place + '/' + 'almanac_params_' + self.tag + '.txt'
        else:
            skycalc_save_fl = self.place + '/' + 'skycalc_params.txt'
            almanac_save_fl = self.place + '/' + 'almanac_params.txt'

        skycalc_params_dict = self.skycalc_params_to_dict()
        almanac_params_dict = self.almanac_params_to_dict()

        with open(skycalc_save_fl, 'w') as fl:
            for key in skycalc_params_dict.keys():
                line = "{}\t:\t{}\n".format(key, str(skycalc_params_dict[key]))
                fl.write(line)

        with open(almanac_save_fl, 'w') as fl:
            for key in almanac_params_dict.keys():
                line = "{}\t:\t{}\n".format(key, str(almanac_params_dict[key]))
                fl.write(line)

        return skycalc_save_fl, almanac_save_fl

    def run_skycalc(self, output_fl):
        '''
        function that calls and runs the eso skycalc cli, then adds output to object

        Parameters
        ----------
        output_fl : string
            name of file to output todo
            must be .fits
        '''

        skycalc_save_fl, almanac_save_fl = self.write_skycalc_params()
        command = "skycalc_cli -i {0} -o {1} -a {2}".format(skycalc_save_fl, output_fl, almanac_save_fl)
        print("RUNNING:", command)
        subprocess.run(command.split(' '))
        print("DONE")

        output_fits = pyfits.open(output_fl)
        lam_sky = output_fits[1].data["lam"]
        flux_sky = output_fits[1].data["flux"]
        trans_sky = output_fits[1].data["trans"]

        # Convert sky flux to W/m^2/um to match existing function
        hc = 1.986446e-25 # h*c (kg*m**3/s**2)
        flux_sky = flux_sky * (np.pi * u.steradian)
        flux_sky = flux_sky * (u.radian.in_units(u.arcsec))**2 * (u.arcsec**2 / u.steradian)
        flux_sky = flux_sky / (lam_sky * 1e-6) * hc / (u.photon / u.watt / u.second)
        flux_sky = flux_sky.value / np.pi

        self.lam = lam_sky
        self.flux = flux_sky
        self.trans = trans_sky

    def __str__(self):
        string = 'Sky Flux: \n---------\n'+\
            '- Airmass : '+"%s" % (self.airmass)+'\n'+\
            '- PWV Mode : '+"%s" % (self.pwv_mode)+'\n'+\
            '- Season  : '+"%s" % (self.season)+'\n'+\
            '- Time of Night  : '+"%s" % (self.time)+' \n'+\
            '- Precipitable Water Vapor (PWV) [mm]  : '+"%s" % (self.pwv)+' \n'+\
            '- Monthly Averaged Solar Flux [sfu=0.01 MJy]  : '+"%s" % (self.msolflux)+' \n'+\
            '- Include Scattered Moonlight?  : '+"%s" % (self.incl_moon)+' \n'+\
            '- Separation of Sun and Moon as seen from Earth [deg] : '+"%s" % (self.moon_sun_sep)+' \n'+\
            '- Separation of Moon and target [deg]  : '+"%s" % (self.moon_target_sep)+' \n'+\
            '- Moon altitude over horizon  : '+"%s" % (self.moon_alt)+' \n'+\
            '- Moon-Earth distance (mean=1)  : '+"%s" % (self.moon_earth_dist)+' \n'+\
            '- Include scattered starlight?  : '+"%s" % (self.incl_starlight)+' \n'+\
            '- Include zodiacal light?  : '+"%s" % (self.incl_zodiacal)+' \n'+\
            '- Heliocentric ecliptic longitude [deg]  : '+"%s" % (self.ecl_lon)+' \n'+\
            '- Ecliptic latitude [deg]  : '+"%s" % (self.ecl_lat)+' \n'+\
            '- Include molecular emission of lower atmosphere?  : '+"%s" % (self.incl_loweratm)+' \n'+\
            '- Include molecular emission of upper atmosphere?  : '+"%s" % (self.incl_upperatm)+' \n'+\
            '- Include upper airglow continuum?  : '+"%s" % (self.incl_airglow)+' \n'+\
            '- Include instrumental thermal radiation?  : '+"%s" % (self.incl_therm)+' \n'+\
            '- Instrumental temperature 1  : '+"%s" % (self.therm_t1)+' \n'+\
            '- Instrumental emmisivity 1  : '+"%s" % (self.therm_e1)+' \n'+\
            '- Instrumental temperature 2  : '+"%s" % (self.therm_t2)+' \n'+\
            '- Instrumental emmisivity 2  : '+"%s" % (self.therm_e2)+' \n'+\
            '- Instrumental temperature 3  : '+"%s" % (self.therm_t3)+' \n'+\
            '- Instrumental emmisivity 3  : '+"%s" % (self.therm_e3)+' \n'+\
            '- Calculation in vacuum or air?  : '+"%s" % (self.vacair)+' \n'+\
            '- Minimum wavelength [nm]  : '+"%s" % (self.wmin)+' \n'+\
            '- Maximum wavelength [nm]  : '+"%s" % (self.wmax)+' \n'+\
            '- Wavelength grid mode  : '+"%s" % (self.wgrid_mode)+' \n'+\
            '- Wavelength sampling dlambda [nm]  : '+"%s" % (self.wdelta)+' \n'+\
            '- User-defined wavelength sampling  : '+"%s" % (self.wgrid_user)+' \n'+\
            '- Spectral resolution  : '+"%s" % (self.wres)+' \n'+\
            '- Line spread function type  : '+"%s" % (self.lsf_type)+' \n'+\
            '- Gaussian FWHM  : '+"%s" % (self.lsf_gauss_fwhm)+' \n'+\
            '- Boxcar FWHM  : '+"%s" % (self.lsf_boxcar_fwhm)+' \n'+\
            '- Observatory  : '+"%s" % (self.observatory)+' \n'+\
            '- Right Ascension  : '+"%s" % (self.ra)+' \n'+\
            '- Declination  : '+"%s" % (self.dec)+' \n'+\
            '- Date  : '+"%s" % (self.date)+''
        return string

if __name__ == '__main__':
    sf = SkyFlux()
    sf.place = './testing/'
    sf.run_skycalc('./output.fits')
    print(sf.lam)
    print(sf.flux)
    print(sf.trans)
