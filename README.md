# coronagraph

A Python noise model for directly imaging exoplanets with a space based coronagraph. 

If you use this model in your own research please cite [Robinson et al (2015)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1507.00777) and include the following acknowledgement: "This work made use of the Python coronagraph noise model, developed by J. Lustig-Yaeger and available at https://github.com/jlustigy/coronagraph/". 

## Install

* Clone this github repository, and the dependent [`readsmart`](https://github.com/jlustigy/readsmart) module within:
```shell
git clone --recursive git@github.com:jlustigy/coronagraph.git
```
* (optional) Add to python path to use `coronagraph` in any dirctory
```shell
# For .cshrc (I know, terrible...)
setenv PYTHONPATH ${PYTHONPATH}:location/of/coronagraph/
```

## Examples

#### Simulate observation with the Integral Field Spectrograph (IFS)

```python
# Import coronagraph package
import coronagraph as cg

# Initialize Telescope, Planet, and Star objects
telescope = cg.Telescope()
planet = cg.Planet()
star = cg.Star()

# Read-in wavelength, reflectance model
model = np.loadtxt('planets/earth_quadrature_radiance_refl.dat', skiprows=8)
lam = model[:,0]            # wavelength (microns)
refl = np.pi * model[:,3]   # geometric albedo
solhr = model[:,2]          # solar flux

# Specify telescope integration time in hours
integration_time = 10.0

# Observe!
lam, dlam, Cratio, spec, sig, SNR = \
      cg.generate_observation(lam, refl, solhr, integration_time, telescope, planet, star)

```
<img src="https://github.com/jlustigy/coronagraph/blob/master/plots/earth_quad_R70.png" width="100%" height="100%" align="middle" />

#### Simulate observation with the Imaging camera

```python
# Set telescope to 'Imaging' mode
telescope.mode = 'Imaging'

# Load Filter Wheel for obsevation (the default filters are the Johnson-Counsins UBVRI filters)
landsat = cg.filters.landsat()
jc = cg.filters.johnson_cousins2()

# Add Filter Wheel to Telescope
telescope.filter_wheel = jc

# Observe!
lam, spec, sig = cg.generate_observation(lam, refl, integration_time, telescope, planet, star)
```
<img src="https://github.com/jlustigy/coronagraph/blob/master/plots/earth_quad_jc.png" width="100%" height="100%" align="middle" />

## Notes

* See [notebooks](https://github.com/jlustigy/coronagraph/tree/master/notebooks) for more examples
* Check out the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=2MASS/2MASS.J&&mode=browse&gname=2MASS&gname2=2MASS#filter)
