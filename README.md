# coronagraph
A Python noise model for directly imaging exoplanets with a space based coronagraph 

# Examples

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

See [notebooks](https://github.com/jlustigy/coronagraph/tree/master/notebooks) for more examples
