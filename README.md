# coronagraph
A Python noise model for directly imaging exoplanets with a space based coronagraph. See [this paper on the arXiv](http://arxiv.org/abs/1507.00777) for a thorough description of the model. 

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

# Set location of SMART .rad file to be "observed"
smart_rad_file = 'planets/F2V_5.e-1fCO2_1.e6H2Volc_1.e10BIF.out_toa.rad'

# Specify telescope integration time
integration_time = 20.0  # hours

# Observe!
lam, spec, sig, wlhr, Ahr = cg.smart_observation(smart_rad_file, integration_time, telescope, planet, star)
```
<img src="https://github.com/jlustigy/coronagraph/blob/master/plots/example1.png" width="100%" height="100%" align="middle" />

### Simulate observation with the Imaging camera

```python
# Set telescope to 'Imaging' mode
telescope.mode = 'Imaging'

# Load Filter Wheel for obsevation (if not the default Johnson-Counsins filters)
landsat = cg.filters.landsat()

# Add Filter Wheel to Telescope
telescope.filter_wheel = landsat

# Observe!
lam, spec, sig, wlhr, Ahr = cg.smart_observation(smart_rad_file, integration_time, telescope, planet, star)
```

See [notebooks](https://github.com/jlustigy/coronagraph/tree/master/notebooks) for more examples
