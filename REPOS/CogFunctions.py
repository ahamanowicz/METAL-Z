import numpy as np
import matplotlib.pyplot as plt
import scipy as scy
from astropy import convolution
import numpy.ma as ma
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from astropy.io import fits

def func(x, beta):
    f = 1 - np.exp(-beta*np.exp(-x**2))
    return f

#return Wl/l
def CoG_full(N,b,lf):
    """
    Curvey of Growth analytical function for given column density, b parameter over a range of LF (wavelength * f-parameter).
    Retunrs and array.
    """
    
    X = np.linspace(0.,1.e3, 1000000)
    alpha = 1.497e-2 #cm2/s
    N = 10 ** N
    b *= 1e5
    lf = 10**lf
    tau = alpha * lf * N/b
    c=3.e10
    W_arr = []
    
    for l in lf:
        tau = alpha * l * N/b
        Y = quad(func,0,1e3, args=(tau))[0]
        W  = 2 * b /c *Y
        W_arr= np.append(W_arr, W)
    
    #print(Y,W)
    
    return W_arr  
    
def CoG_single(N,b,lf):
    
    """
    Curvey of Growth analytical function for given column density, b parameter at a single wavelength.
    Returns single value
    """
    
    X = np.linspace(0.,1.e3, 1000000)
    alpha = 1.497e-2 #cm2/s
    N = 10**N
    b*=1e5
    lf = 10**lf
    tau = alpha * lf * N/b
    c=3.e10

    Y = quad(func,0,1e3, args=(tau))[0]
    W  = 2 * b /c *Y
    #print(Y,W)
    
    return W  

def banana_plot(Chi_r, name='', line='', n_voigt=0, contours=[0.99, 0.98,0.95,0.90, 0.7, 0.5, 0.2], xlim = [14,17], ylim=[5,25], save=True):

    """
    plot the error contours of CoG analysis
    name - name of the sightline,
    line - name of the ion 
    Chi_r - map of the chi^2 
    n_voigt - logN measurement from voigt fitting. Default is 0, so no plot
    L - probability map for plotting
    contours -levels you want displayed on the plot
    xlim -  limits in log (N)
    ylim - limits in b [km/s]
    
    """
    fig = plt.figure()
    min_chi = np.min(Chi_r)
    L = np.exp(-Chi_r/2.) #translate chi map to probability map for plotting
    Lmax = np.max(L) 
    
    ## define contours
    lev = 1-np.array(contours) #contour levels 9probability)
    strs = np.array(contours, dtype='str') #contour descriptions
    
    CS = plt.contour(L.T, levels=lev*Lmax, extent=(10,20,5,100),colors='k') #define contours
    
    #substitute generic contour labels with desired
    fmt={}
    
    for l,s in zip( CS.levels, strs ):
        fmt[l] = s
        
    cb = plt.clabel(CS,inline=1,inline_spacing=-5,fontsize=12,fmt=fmt,colors='b') #contour labels
    
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel("log N")
    plt.ylabel("b (km/s)")

    plt.annotate(r"$\chi^2_{min}$ = "+str(round(min_chi,2)) , (0.1,0.1), xycoords='axes fraction')
    plt.annotate(name + " " + line, (0.6,0.9), xycoords='axes fraction')

    if n_voigt > 0:
        plt.axvline(n_voigt, ls='--', c='maroon')

    #save the L arry to the file to recreate banana plots
    np.savetxt(name+"_"+line+"_L.txt", L)
    fig.savefig(name+"_"+line+"_cog.png")
        
def cog_model(Narray=np.arange(10,20,0.1), barray=np.arange(5,100,.5) , LFarray=np.linspace(-8,-4,100)):
    """
    Generate the model array
    """
    Model_array=np.zeros((np.size(N),np.size(b), np.size(LF_m)))
    for i in range(np.size(N)):
        for j in range(np.size(b)):
            Model_array[i][j] = np.log10(cog.CoG_full(N[i],b[j],LF_m))
    return Model_array
