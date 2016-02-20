# coronagraph
A Python noise model for directly imaging exoplanets with a space based coronagraph 

----

# Examples
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
