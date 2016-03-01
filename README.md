# coronagraph
A Python noise model for directly imaging exoplanets with a space based coronagraph. See [Robinson et al (2015)](http://arxiv.org/abs/1507.00777) for a thorough description of the model. 

# Examples

### Simulate observation with the Integral Field Spectrograph (IFS)

Use coronagraph noise model to generate an observation of high-resolution SMART output:
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

# Specify telescope integration time in hours
integration_time = 10.0

# Observe!
lam, spec, sig = cg.generate_observation(lam, refl, integration_time, telescope, planet, star)
```
<img src="https://github.com/jlustigy/coronagraph/blob/master/plots/earth_quad_R70.png" width="100%" height="100%" align="middle" />

### Simulate observation with the Imaging camera

```python
# Set telescope to 'Imaging' mode
telescope.mode = 'Imaging'

# Load Filter Wheel for obsevation (if not the default Johnson-Counsins filters)
landsat = cg.filters.landsat()
jc = cg.filters.johnson_cousins2()

# Add Filter Wheel to Telescope
telescope.filter_wheel = jc

# Observe!
lam, spec, sig = cg.generate_observation(lam, refl, integration_time, telescope, planet, star)
```
<img src="https://github.com/jlustigy/coronagraph/blob/master/plots/earth_quad_jc.png" width="100%" height="100%" align="middle" />


See [notebooks](https://github.com/jlustigy/coronagraph/tree/master/notebooks) for more examples
