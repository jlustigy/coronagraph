import coronagraph as cg
jc = cg.imager.johnson_cousins()
import matplotlib.pyplot as plt
cg.plot_setup.setup()
jc.plot(ylim = (0.0, 1.2))
plt.show()