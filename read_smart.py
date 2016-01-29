
import numpy as np

def read_rad(path):
    rad_data = np.genfromtxt(path, skiprows=0)
    wl = rad_data[:,0]
    wno = rad_data[:,1]
    solar_spec = rad_data[:,2]
    TOA_flux = rad_data[:,3] #W/m**2/um
    rad_streams = rad_data[:,4::]
    return wl, wno, solar_spec, TOA_flux, rad_streams

def get_jacobians(path,jext,stream = 1):	

    '''Reads-in and parses SMART jacobians, returns a set of numpy arrays.
     
    Parameters
    ----------
    path : string
        SMART radiance output file
        Ex: 'earth_standard_hitran08_50_100000cm-1_clearsky_toa.rad'
    jext : string
        A unique indentifier used as the file extension for each Jacobian output --
        Specifies a physical quantity for which Jacobians have been calculated.
        Ex: '.j_O3'
    stream : int
        Specifices the upwelling stream to read-in

    Returns
    ----------
    jwl : float64
        Wavelength grid
    jwno : float64
        Wavenumber grid
    presprofile: float64
        Pressure grid
    jrad : float64
        Upwelling radiance grid 
    jacobians : float64
        Jacobian grid
    vprofile : float64
        Profile (or single quantity) corresponding to unperturbed state
    jfraction : float64
        Fractional change for which Jacobians were computed 

    Revision History
    ----------
    Written by J. Lustig-Yaeger June, 2015

    Examples    
    ----------  
    In [1]: from read_jacobians import get_jacobians
    In [2]: jext = '.j_O2'
    In [3]: dpath = '/astro/users/jlustigy/Models/smart_training/case06/'
    In [4]: radf = 'earth_standard_clearsky_jacobians_0.01_toa.rad'
    In [5]: jpath = dpath+radf
    In [6]: jwl, jwno, jpres, jrad, jacobians, vprofile, jfrac = get_jacobians(jpath,jext,stream=1)

    '''

    jfiles = ['_rad001_toa','_rad002_toa','_rad003_toa','_rad004_toa']
    jpath1 = path+jfiles[0]+jext
    jpath2 = path+jfiles[1]+jext
    jpath3 = path+jfiles[2]+jext
    jpath4 = path+jfiles[3]+jext
    jpaths = [jpath1,jpath2,jpath3,jpath4]
    
    if jext == '.j_pres' or jext == '.j_surf':
        single_stream = single_stream2
    else:
        single_stream = single_stream1
    
    if stream != 1 and stream != 2 and stream != 3 and stream != 4 and stream !='all':
        print('Invalid Stream. Must be 1, 2, 3, 4, or "all"')
        return
    elif stream == 1 or stream == 2 or stream == 3 or stream == 4:
        jwl, jwno, presprofile, jrad, jacobians, vprofile, jfraction \
        = single_stream(jpaths[stream-1])
    elif stream == 'all':
        jwl1, jwno1, presprofile1, jrad1, jacobians1, vprofile1, jfraction1 = \
        single_stream(jpath1)
        jwl2, jwno2, presprofile2, jrad2, jacobians2, vprofile2, jfraction2 = \
        single_stream(jpath2)
        jwl3, jwno3, presprofile3, jrad3, jacobians3, vprofile3, jfraction3 = \
        single_stream(jpath3)
        jwl4, jwno4, presprofile4, jrad4, jacobians4, vprofile4, jfraction4 = \
        single_stream(jpath4)
        
        jwl = np.array([jwl1,jwl2,jwl3,jwl4])
        jwno = np.array([jwno1,jwno2,jwno3,jwno4])
        presprofile = np.array([presprofile1,presprofile2,presprofile3,presprofile4])
        jrad = np.array([jrad1,jrad2,jrad3,jrad4])
        jacobians = np.array([jacobians1,jacobians2,jacobians3,jacobians4])
        vprofile = np.array([vprofile1,vprofile2,vprofile3,vprofile4])
        jfrac = jfraction
    
    print('Jacobian shape: '+str(jacobians.shape))
    print('Initial values: ')
    print(vprofile) 
        
    return jwl, jwno, presprofile, jrad, jacobians, vprofile, jfraction

    
def read_header(arr):
    arr[0:30]
    layers = arr[0][0]
    jfraction = arr[1][0]
    vprofile = arr[1][1:]
    i=2
    while len(vprofile) < layers:
        vprofile = np.concatenate([vprofile,arr[i]])
        i += 1
    wnorange = [arr[i][0],arr[i][1]]
    presprofile = arr[i][2:]
    i += 1
    while len(presprofile) < layers:
        presprofile = np.concatenate([presprofile,arr[i]])
        i += 1
    return i, layers, jfraction, vprofile, wnorange, presprofile

def read_header2(arr):
    layers = arr[0][0]
    jfraction = arr[1][0]
    jquant = arr[1][1]
    i=2

    wnorange = [arr[i][0],arr[i][1]]
    presprofile = arr[i][2:]
    i += 1
    while len(presprofile) < layers:
        presprofile = np.concatenate([presprofile,arr[i]])
        i += 1
    return i, layers, jfraction, jquant, wnorange, presprofile

def parse_jacobians(arr):
    
    # Extract info from header
    index, layers, jfraction, vprofile, wnorange, presprofile \
    = read_header(arr)
    
    # Define a bunch of quantities from header of file
    x = len(arr)*1.0
    y = len(arr[index])*1.0
    p_tot = len(presprofile)*1.0
    z = np.mod(p_tot+3, y)*1.0
    a = np.floor((p_tot+3)/y)
    
    # if there is an additional unfilled row then d=1, else d=0
    if z == 0: 
        d = 0*1.0
    else:
        d = 1.0
    
    h = int(a+d) # rows in file per true row
    N = (x-index)/(a+d) # total number of true rows in file
    wno1 = arr[index][1]*1.0
    wno2 = arr[index+int(a+d)][1]*1.0
    ranwno = (wnorange[1] - wnorange[0])*1.0
    deltawno = (wno2 - wno1)*1.0
    wnotot = ranwno / deltawno
    
    # Construct structured array for data (N, layers+3)
    J = np.zeros((N,p_tot+3))
    
    # loop over all N in J, filling true row with appropriate file rows
    for i in range(int(N)):
        try:
            J[i,:] = np.hstack(arr[(h*i)+index:(h*(i+1)+index)])
        except IndexError:
            print("IndexError: i = "+str(i), J.shape) 
            break
    
    # decompose J into discrete pieces
    jwl, jwno, jrad, jacobians = J[:,0], J[:,1], J[:,2], J[:,3:]
    
    return jwl, jwno, presprofile, jrad, jacobians, vprofile, jfraction

def parse_jacobians2(arr):
    
    # Extract info from header
    index, layers, jfraction, jquant, wnorange, presprofile \
    = read_header2(arr)
        
    # Construct structured array for data 
    J = np.zeros((len(arr[index:]),4))
    Jarr= np.array(arr[index:])
    
    # loop over all indexes assigning each row to J
    for i in range(len(Jarr)):
        J[i,0] = Jarr[i][0]
        J[i,1] = Jarr[i][1]
        J[i,2] = Jarr[i][2]
        J[i,3] = Jarr[i][3]

    jwl, jwno, jrad, jacobians = J[:,0], J[:,1], J[:,2], J[:,3]
    
    return jwl, jwno, presprofile, jrad, jacobians, jquant, jfraction

def single_stream1(path):
    
    print("Opening j_file: "+path)
    arrays = [np.array(map(float, line.split())) for line in open(path)]
    arrays = np.array(arrays)
    
    print("Length of file: "+str(len(arrays)))
    
    jwl1, jwno1, presprofile1, jrad1, jacobians1, vprofile1, jfraction1 = \
    parse_jacobians(arrays)
    
    return jwl1, jwno1, presprofile1, jrad1, jacobians1, vprofile1, jfraction1

def single_stream2(path):
    
    print("Opening j_file: "+path)
    arrays = [np.array(map(float, line.split())) for line in open(path)]
    arrays = np.array(arrays)
    
    print("Length of file: "+str(len(arrays)))
    
    jwl1, jwno1, presprofile1, jrad1, jacobians1, jquant1, jfraction1 = \
    parse_jacobians2(arrays)
    
    return jwl1, jwno1, presprofile1, jrad1, jacobians1, jquant1, jfraction1
