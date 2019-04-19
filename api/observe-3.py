from coronagraph.observe import planetzoo_observation
from coronagraph import plot_setup
plot_setup.setup()
lam, spec, sig = planetzoo_observation(name = "mars", plot = True)