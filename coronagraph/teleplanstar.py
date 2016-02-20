import numpy as np

'''
Contains the Telescope, Planet, and Star classes.
'''

class Telescope(object):
    
    '''
    Parameters
    ----------
    
    lammin : float
        Minimum wavelength (um)
    lammax : float 
        Maximum wavelength (um)
    R : float 
        Spectral resolution (lambda / delta-lambda)
    Tsys : float 
        Telescope temperature (K)
    D : float
        Telescope diameter (m) 
    emis : float 
        Telescope emissivity
    IWA : float
        Inner Working Angle (lambda/D)
    OWA : float 
        Outer Working Angle (lambda/D)
    Tput : float 
        Telescope throughput   
    C : float 
        Raw Contrast
    De : float 
        Dark current (s**-1)
    DNHpix : float
        Horizontal pixel spread of IFS spectrum
    Re : float 
        Read noise per pixel
    Dtmax : float
        Maximum exposure time (hr)
    X : float
        Size of photometric aperture (lambda/D)
    q : float
        Quantum efficiency
    '''
    
    # Define a constructor
    def __init__(self, lammin=0.3,lammax=2.0,R=70.,Tput=0.2,\
                 D=8.0,Tsys=274.,IWA=0.5, OWA=30000.,emis=0.9,\
                 C=1e-10,De=1e-4,DNHpix=3.,Re=0.1,Dtmax=1.0,X=0.7,q=0.9):
        
        self.lammin=lammin
        self.lammax=lammax
        self.resolution=R
        self.throughput=Tput
        self.diameter=D
        self.temperature=Tsys
        self.IWA=IWA
        self.OWA=OWA
        self.emissivity=emis
        self.contrast=C
        
        self.darkcurrent=De
        self.DNHpix=DNHpix
        self.readnoise=Re
        self.Dtmax=Dtmax
        self.X=X
        self.qe=q
    
    def __str__(self):
        string = 'Coronagraph: \n------------\n'+\
            '- Minimum wavelength (um) : '+"%s" % (self.lammin)+'\n'+\
            '- Maximum wavelength (um)  : '+"%s" % (self.lammax)+'\n'+\
            '- Spectral resolution (lambda / delta-lambda)  : '+"%s" % (self.resolution)+' \n'+\
            '- Telescope temperature (K)  : '+"%s" % (self.temperature)+' \n'+\
            '- Telescope diameter (m)  : '+"%s" % (self.diameter)+' \n'+\
            '- Telescope emissivity  : '+"%s" % (self.emissivity)+' \n'+\
            '- Inner Working Angle (lambda/D)  : '+"%s" % (self.IWA)+' \n'+\
            '- Outer Working Angle (lambda/D)  : '+"%s" % (self.OWA)+' \n'+\
            '- Telescope throughput  : '+"%s" % (self.throughput)+' \n'+\
            '- Raw Contrast  : '+"%s" % (self.contrast)+' \n'+\
            '- Dark current (s**-1)  : '+"%s" % (self.darkcurrent)+' \n'+\
            '- Horizontal pixel spread of IFS spectrum  : '+"%s" % (self.DNHpix)+' \n'+\
            '- Read noise per pixel  : '+"%s" % (self.readnoise)+' \n'+\
            '- Maximum exposure time (hr)  : '+"%s" % (self.Dtmax)+' \n'+\
            '- Size of photometric aperture (lambda/D)  : '+"%s" % (self.X)+' \n'+\
            '- Quantum efficiency  : '+"%s" % (self.qe)+''
        return string

def lambertPhaseFunction(alpha):
    '''Calculate the Lambertian Phase Function from the phase angle.
    Args:
        alpha: Planet phase angle (degrees)
    Returns:
        The Lambertian phase function
    '''
    alpha = alpha * np.pi / 180.
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi

class Planet(object):
    '''Parameters of the planet to be observed.
    
    Parameters
    ----------
    
    name : string
        Planet name from database
    star : string
        Stellar type of planet host star
    d : float
        Distance to system (pc)
    Nez : float
        Number of exzodis (zodis)
    Rp : float
        Radius of planet (Earth Radii)
    a : float
        Semi-major axis (AU)
    alpha : float
        Phase angle (deg)
    Phi : float
        Lambertian phase function
    MzV : float
        Zodiacal light surface brightness (mag/arcsec**2)
    MezV : float
        exozodiacal light surface brightness (mag/arcsec**2)
    '''
    
    # Define a constructor
    def __init__(self, name='earth', star='sun', d=10.0,Nez=1.0,\
                 Rp=1.0, a=1.0, alpha=90.,\
                 MzV=23.0, MezV=22.0):
        self.name=name
        self.star=star
        self.distance=d
        self.Nez=Nez
        self.Rp=Rp
        self.a=a
        self._alpha=alpha
        self._Phi = None
        self.MzV  = MzV     # zodiacal light surface brightness (mag/arcsec**2)
        self.MezV = MezV     # exozodiacal light surface brightness (mag/arcsec**2)
        
        if self._Phi is None:
            self._Phi = lambertPhaseFunction(self._alpha)
        else:
            raise Exception("Error in Planet Phase Function (Phi)")
  
    @property
    def alpha(self):
        return self._alpha
  
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self._Phi = lambertPhaseFunction(value)
  
    @property
    def Phi(self):
        return self._Phi
  
    @Phi.setter
    def Phi(self, value):
        self._Phi = value

    def __str__(self):
        string = 'Planet: \n-------\n'+\
            '- Planet name  : '+"%s" % (self.name)+'\n'+\
            '- Stellar type of planet host star : '+"%s" % (self.star)+'\n'+\
            '- Distance to system (pc) : '+"%s" % (self.distance)+'\n'+\
            '- Number of exzodis (zodis) : '+"%s" % (self.Nez)+'\n'+\
            '- Radius of planet (Earth Radii) : '+"%s" % (self.Rp)+'\n'+\
            '- Semi-major axis (AU) : '+"%s" % (self.a)+'\n'+\
            '- Phase angle (deg) : '+"%s" % (self.alpha)+'\n'+\
            '- Lambertian phase function : '+"%s" % (self.Phi)+'\n'+\
            '- Zodiacal light surface brightness (mag/arcsec**2) : '+"%s" % (self.MzV)+'\n'+\
            '- Exozodiacal light surface brightness (mag/arcsec**2) : '+"%s" % (self.MezV)
        return string
            
class Star(object):
    
    def __init__(self, Teff=5780.0, Rs=1.0):
        self.Teff=Teff
        self.Rs=Rs
    
    def __str__(self):
        string = 'Star: \n-----\n'+\
            '- Effective Temperature (K) : '+"%s" % (self.Teff)+'\n'+\
            '- Radius (Solar Radii) : '+"%s" % (self.Rs)
        return string