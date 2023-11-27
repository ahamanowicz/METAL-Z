import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import glob
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from astropy.convolution import convolve, Box1DKernel
from astropy.coordinates import SkyCoord
import math
import os
import matplotlib
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import math
import sys
sys.path.append("/astro/dust_kg/jduval/py_utils")
from numerics import fwhm, get_percentiles #get_p50
import scipy
#from skimage.morphology import watershed
#from skimage.filters import sobel
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.interpolate import UnivariateSpline
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from spectres import spectres
from scipy.stats import pearsonr
import astropy.units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from scipy.interpolate import *
from scipy.integrate import trapezoid, cumulative_trapezoid
from foregrounds import get_mw_nhi21cm




def cog_function(logN, logflambda, b):
    """
    logN in cm-2
    wave in Angstroms
    logflamba in A
    b in km/s
    """


    me = 9.10938370*1e-28
    e=4.80320427* 1.e-10
    c=2.99792458*1.e10
    tau0 = np.sqrt(np.pi)*e**2/me/c* 10.**(logflambda)*1e-8*10.**(logN)/(b*1e5)

    print("TAU 0", tau0)

    x = np.concatenate((np.array([0]), 10.**(np.arange(-6, 6, 0.05))))

    ftau0 = (1.-np.exp(-tau0*np.exp(-x**2)))
    Wol = 2.*(b*1.e5)*np.trapz(ftau0, x)/c
    return(Wol)

def cog_analysis(ews_over_lambda, err_ews, lambdas, logflambdas, logN,err_logN ,b=30., title = ''):

    fig = plt.figure(figsize = (10,8))

    plt.errorbar(logflambdas, np.log10(ews_over_lambda), yerr = err_ews/ews_over_lambda/np.log(10.), fmt ='ro', markersize = 20)

    mod_logflambdas = np.arange(-1, 4, 0.2)
    mod_ews = np.zeros(len(mod_logflambdas))
    mod_ews_minus = np.zeros(len(mod_logflambdas))
    mod_ews_plus = np.zeros(len(mod_logflambdas))

    for i in range(len(mod_ews)):
        mod_ews[i] = cog_function(logN, mod_logflambdas[i], b)
        mod_ews_minus[i] = cog_function(logN-err_logN, mod_logflambdas[i], b)
        mod_ews_plus[i] = cog_function(logN + err_logN, mod_logflambdas[i], b)
    plt.plot(mod_logflambdas, np.log10(mod_ews),  'k-')
    plt.plot(mod_logflambdas, np.log10(mod_ews_minus),  'k--')
    plt.plot(mod_logflambdas, np.log10(mod_ews_plus),  'k--')
    plt.xlabel("log f"+r'$\lambda$', fontsize = 23)
    plt.ylabel(r'W$_{\lambda}$' + "/"+r'$\lambda$', fontsize = 23)
    plt.title(title, fontsize = 23)

    plt.tight_layout()
    plt.show()




def run_mySTIS_merge_list(files):

    for i in range(len(files)):
        outfile = files[i].replace('_x1d.fits', '_x1_merged.fits')
        mySTIS_merge_v1(outfile, stis_file = files[i])

def mySTIS_merge_v1(outfile, w_in=[],f_in = [],e_in=[],b_in = [], stis_file = ''):

    if stis_file !='':

        s = fits.open(stis_file)
        s= s[1].data
        w = s['WAVELENGTH']
        f = s['FLUX']
        e  = s['ERROR']
        b = s['BACKGROUND']
    else:
        w = w_in
        f = f_in
        e = e_in
        b = b_in

    sh = w.shape
    nord = sh[0]
    nw = sh[1]

    w = w[:, 9:nw-10]
    f = f[:, 9:nw-10]
    e = e[:, 9:nw-10]
    b = b[:, 9:nw-10]
    nw = w.shape[1]

    wmin = np.nanmin(w)
    wmax = np.nanmax(w)
    wdel = np.abs(np.median(w-np.roll(w,1)))
    new_w = np.arange(wmin, wmax, wdel)

    coadd_files = []

    for i in range(nord):
        t = Table()
        t['WAVELENGTH'] = w[i,:]
        t['FLUX']= f[i,:]
        t['ERROR'] = e[i,:]
        ascii.write(t,'spec_order_{}.dat'.format(i), overwrite=True)
        coadd_files.append('spec_order_{}.dat'.format(i))

    coadd_files = np.array(coadd_files)
    coadd_fluxes(coadd_files, writefits = outfile, new_wavelength = new_w)

    for i in range(nord):
        os.remove(coadd_files[i])





def read_basic_spectrum(file, dq_col = ''):


    if '.fit' in file:
        spec = fits.open(file)
        spec = spec[1].data

        if 'WAVE' in spec.columns.names:
            w = spec['WAVE'].flatten()
        else:
            w = spec['WAVELENGTH'].flatten()

        f = spec['FLUX'].flatten()

        if 'ERROR' in spec.columns.names:
            e = spec['ERROR'].flatten()
        else:
            e  = np.zeros_like(f)

        if dq_col != '':
            dq = spec[dq_col].flatten()

    else:

        spec = ascii.read(file)

        if 'WAVELENGTH' in spec.colnames:

            w = spec['WAVELENGTH'].data
            f = spec['FLUX'].data
            e = spec['ERROR'].data
        else:
            w= spec['col1'].data
            f = spec['col2'].data
            e = spec['col3'].data
        if dq_col != '':
            dq = spec[dq_col].data

    indsort = np.argsort(w)
    w = w[indsort]
    f = f[indsort]
    e = e[indsort]

    if dq_col != '':
        dq= dq[indsort]

    if dq_col != '':
        return(w,f,e,dq)
    else:
        return(w,f,e)



def coadd_fluxes(spec_files, new_wavelength = [], dq_col = '', writefits='', debug=False):


    """
    Co-adds spectra in spec_files (either ascii or fits). If a DQ_WGT needs to be applied, specify name of tbat column in dq_col keyword (a string, e.g., DQ_WGT for COS)
    This only works for relatively bright fluxes. If fluxes are low and quantized, counts need to be co-added
    """


    nspec = len(spec_files)

    #first determine the new wavelength grid if not provided, including the shortest and longest wavelengths if final wavelength array is not supplied, and the longest dispersion

    if len(new_wavelength)==0:

        min_w = 1.e8
        max_w = 0.
        disp = 0.

        for i in range(nspec):
            w,f,e = read_basic_spectrum(spec_files[i])
            this_disp  = np.abs(np.median(w-np.roll(w,1)))
            if np.nanmin(w) < min_w:
                min_w =np.nanmin(w)
            if np.nanmax(w)>max_w:
                max_w = np.nanmax(w)
            if this_disp > disp:
                disp = this_disp

        wgrid = np.arange(min_w,max_w,disp, dtype = 'float64')
    else:
        wgrid = new_wavelength

    print("NEW W ",  wgrid.shape, np.abs(np.median(wgrid-np.roll(wgrid,1))))
    #now proceed with the interpolation and coadd

    fcoadd = np.zeros(len(wgrid), dtype = 'float64')
    ecoadd = np.zeros(len(wgrid), dtype = 'float64')
    weights = np.zeros(len(wgrid), dtype = 'float64')


    for i in range(nspec):
        if dq_col !='':
            w,f,e,dq_wgt= read_basic_spectrum(spec_files[i], dq_col = dq_col)
            ind = np.where(dq_wgt >1.)
            dq_wgt[ind] = 1.

        else:
            w,f,e= read_basic_spectrum(spec_files[i], dq_col = '')
            dq_wgt = np.zeros_like(w) + 1.

        fi_func = interpolate.interp1d(w,f, kind = 'nearest', fill_value = np.nan, bounds_error=False)
        fi = fi_func(wgrid)
        ei_func = interpolate.interp1d(w,e, kind = 'nearest', fill_value = np.nan, bounds_error=False)
        ei = ei_func(wgrid)
        dqi_func = interpolate.interp1d(w,dq_wgt, kind = 'nearest', fill_value = 0., bounds_error=False)
        dqi = dqi_func(wgrid)


        good = np.where((dqi >0.))

        this_ei =np.double( ei*1.e13)
        fcoadd[good] = fcoadd[good] + np.double(fi[good])*1./this_ei[good]**2
        ecoadd[good] = ecoadd[good] + 1./this_ei[good]**2 #error squared for now
        weights[good] = weights[good]   + 1./this_ei[good]**2


        if debug==True:

            print(fcoadd.shape, weights.shape)
            #plt.plot(wgrid, fcoadd/weights, 'k-', alpha = 0.5)
            #plt.plot(wgrid,fi, 'r-', alpha = 0.5)
            #g = np.where(dq_wgt >0)
            #plt.plot(w[g],f[g], 'g.', alpha = 0.5)

            #plt.plot(wgrid[good], fi[good], 'k.', alpha = 0.5)
            #plt.plot(wgrid[good], ei[good], 'r.', alpha =0.5)
            #plt.plot(wgrid,fi, 'r.', alpha  = 0.5)

            #plt.plot(wgrid, weights, 'k.')
            #plt.show()


    fcoadd = fcoadd/weights
    ecoadd = np.sqrt(ecoadd)*1.e-13/weights

    if writefits != '':
        c1 = fits.Column(name = 'WAVELENGTH', array = wgrid, format= 'E')
        c2 = fits.Column(name = 'FLUX', array = fcoadd, format = 'E')
        c3 = fits.Column(name = 'ERROR', array = ecoadd, format = 'E')

        columns = fits.ColDefs([c1, c2, c3])
        hdu = fits.BinTableHDU.from_columns(columns)
        hdu.writeto(writefits, overwrite=True)


    return(wgrid, fcoadd, ecoadd)



def plot_spectrum(spec_files, smooth=1):

    for i in range(len(spec_files)):
        spec_file = spec_files[i]
        if ".fits" in spec_file:
            spec = fits.open(spec_file)
            hdr0 = spec[0].header
            hdr1 = spec[1].header
            s= spec[1].data

            if 'WAVELENGTH' in spec[1].columns.names:
                w = s['WAVELENGTH'].flatten()
            else:
                w = s['WAVE'].flatten()
            f = s['FLUX'].flatten()

            if 'DQ_WGT' in spec[1].columns.names:
                dq = s['DQ_WGT'].flatten()
            else:
                dq = np.zeros_like(f) + 1.

            if smooth >1:
                f = convolve(f, Box1DKernel(smooth))
            #else:
            #    spec = ascii.open(spec_file)
            #    if 'wavelength' in spec.columns


            #plt.plot(w,f*dq, '-', label = hdr0['TARGNAME'] + '/' + hdr0['INSTRUME']+'/' + '{}'.format(hdr0['CENWAVE']) +'-' + '{:4.0f}'.format(hdr1['EXPTIME']))
            plt.plot(w,f*dq, '-', label = spec_files[i])

            plt.xlabel("Wavelength", fontsize = 18)
            plt.ylabel("Flux", fontsize = 18)
    plt.legend()
    plt.show()







def calculate_coeff(a, b, shift):
    # for correlation coefficent
    if shift > 0:
        a = a[shift:]
        b = b[:-shift]
    elif shift < 0:
        a = a[:shift]
        b = b[-shift:]
    # else if shift == 0 then we just use a and b as is
    corr_coeff = abs(pearsonr(a, b)[0])

    return corr_coeff

def quad_fit(c, minpix=5, maxpix=5):

    if len(c) == 1:
        return None
    x = np.arange(len(c))
    if np.argmax(c)-minpix > 0:
        x = x[np.argmax(c)-minpix : np.argmax(c)+(maxpix+1)]
        c2 = c[np.argmax(c)-minpix : np.argmax(c)+(maxpix+1)]
    else:
        x = x[0 : np.argmax(c)+(maxpix+1)]
        c2 = c[0 : np.argmax(c)+(maxpix+1)]
    try:
        quad_fit_func = np.poly1d(np.polyfit(x, c2, 2))
        new_shift = (-quad_fit_func[1]/(2*quad_fit_func[2])) # zero point -b/2a
    except ValueError:
        import pdb; pdb.set_trace()

    #plt.plot(x, quad_fit_func(x))
    #plt.plot(new_shift, np.max(c), '^', color='blue', alpha=0.5)

    return new_shift



def direct_correlate(a, b, fit_peak):

    # direct correlation
    c = np.correlate(a, b, mode='full')
    shift = np.argmax(c)

    #if np.isnan(c).any():
    #    return(None, None, None, None)
    if fit_peak:
        shift = quad_fit(c)
    #if shift == None:
    #    return(None, None, None, None)

    er = abs(shift - np.argmax(c))
    shift = shift - (len(a)-1)
    corr_coeff = calculate_coeff(a, b, int(round(shift)))

    return(shift, c, corr_coeff, er)



def get_hi_hi4pi_spectrum(ra_in, dec_in):

    if isinstance(ra_in, str):
        c = SkyCoord(ra_in, dec_in, frame = 'icrs')
        ra = c.ra.deg
        dec = c.dec.deg
    else:
        ra = ra_in
        dec = dec_in

    coords = SkyCoord(ra*u.deg, dec*u.deg, frame = 'icrs')

    t = ascii.read('/astro/dust_kg/jduval/GASS+EBHIS/cubes_eq_tan.dat')
    rac = t['rac'].data
    decc = t['decc'].data

    dist = np.sqrt((ra-rac)**2 + (dec-decc)**2)
    ind = np.argmin(dist)

    hi_file = t['file'].data[ind]

    print("HI FILE ", hi_file)

    cube = fits.open("/astro/dust_kg/jduval/GASS+EBHIS/" +hi_file)

    wcs = WCS(cube[0].header)
    x ,y, z = wcs.all_world2pix(ra, dec, 0., 0.)


    spec_cube = SpectralCube.read("/astro/dust_kg/jduval/GASS+EBHIS/"+ hi_file)
    velocity = spec_cube.spectral_axis*u.km/u.m
    velocity = np.array(velocity/1000.)

    spectrum = cube[0].data[:, int(np.round(y)), int(np.round(x))]

    return(velocity, spectrum)


def plot_spectral_stack_indiv(w, f_in,e_in, ra, dec,target, galaxy, lines, labels, plotname,  vmin = -300, vmax = 600, vsys = 262., vel_comps = [], smooth = 3):


    if smooth >1 :
        f = convolve(f_in, Box1DKernel(smooth))
        e = convolve(e_in, Box1DKernel(smooth))
        e = e/np.sqrt(smooth)
    else:
        f = f_fin
        e= e_in



    vel_hi, t_hi = get_hi_hi4pi_spectrum(ra, dec)
    #plt.plot(vel_hi, t_hi)
    #plt.show()


    #get the limits of GAL and MW in HI;

    #get the RMS on the HI and find the boundaries of MW and LMC
    indn = np.where((vel_hi < vmin) & (np.isnan(t_hi)==False))
    indn = indn[0]
    rmshi = np.std(t_hi[indn])

    #now just keep what we want
    valid_hi = np.where((vel_hi > vmin) & (vel_hi < vmax))
    vel_hi = vel_hi[valid_hi[0]]
    t_hi = t_hi[valid_hi[0]]

    #identify MW and GAL velocities
    #the right MW boundary is the lowest index above  0 km/s where the line falls under the noise (3sigma)

    if vsys >0:
        ivg_ind = np.where((vel_hi > 10.) & (t_hi < 3.*rmshi) & (vel_hi < vsys-50.))
        ivg_ind = ivg_ind[0]
        vmin_mw = vel_hi[max(ivg_ind)]
        vmax_mw = vel_hi[min(ivg_ind)]
        hvc_color = 'blue'
        mw_color = 'magenta'
        gal_color = 'darkorange'
    else:
        ivg_ind = np.where((vel_hi<-20.) & (t_hi < 3.*rmshi)  & (vel_hi>vsys + 50.))
        ivg_ind= ivg_ind[0]
        if len(ivg_ind)>0:
            vmin_mw = vel_hi[max(ivg_ind)]
            vmax_mw = 50.
        else:
            vmin_mw = -50.
            vmax_mw = 50
        gal_color = 'blue'
        mw_color = 'magenta'
        hvc_color = 'darkorange'



    #vmax_mw = 35.

    print(rmshi)
    gal_ind = np.where((vel_hi > vsys-100.) & (t_hi > 4.*rmshi) & (vel_hi < vsys + 50.))
    gal_ind = gal_ind[0]
    vmin_gal = vel_hi[min(gal_ind)]
    vmax_gal  = vel_hi[max(gal_ind)]

    print("VELOCITIES ", vmin_mw, vmax_mw, vmin_gal, vmax_gal)

    valid = np.where(np.isnan(f)==False & (f > 0.))
    valid= valid[0]


    fig, ax = plt.subplots(nrows=len(lines)+1, ncols = 1, figsize=(6,15.*len(lines)/8.), sharex =True)

    ax[0].plot(vel_hi, t_hi, 'k')
    #ax[0].axvline(x = 262.2, color = 'k', linestyle = '--')

    ax[0].set_xlim(left = vmin, right = vmax)
    ax[0].set_ylabel("T(HI) [K]", fontsize = 16)
    ax[0].set_yscale("log")
    ax[0].axvline(x = vmin_mw, color = 'k', linestyle = '--')
    ax[0].axvline(x = vmax_mw, color = 'k', linestyle = '--')
    ax[0].axvline(x = vmin_gal, color = 'k', linestyle = '--')
    ax[0].axvline(x = vmax_gal, color = 'k', linestyle = '--')
    ax[0].set_ylim(bottom=0.001, top = 1000)
    ax[0].text(0., 100, 'MW', color = mw_color, fontsize = 15)
    if vsys >0:
        ax[0].text(vmax_mw+20., 100, 'I/HVC', color =hvc_color, fontsize = 15)

    ax[0].text(vmin_gal+50., 100, galaxy.upper(), color = gal_color, fontsize = 15)


    ax[0].set_title(target)

    for l in range(len(lines)):


        if (lines[l] > min(w[valid]) and (lines[l] < max(w[valid]))):


            velocity = (w-lines[l])/lines[l]*3.e5
            velocity = helio_to_lsr(velocity, ra, dec)

            if vsys>0:
                ind_mw = np.where((velocity> vmin) & (velocity < vmax_mw))
                ind_mw = ind_mw[0]
                ind_ivg = np.where((velocity > vmax_mw) & (velocity < vmin_gal))
                ind_ivg = ind_ivg[0]

                ind_gal  = np.where((velocity > vmin_gal) & (velocity < vmax))
                ind_gal = ind_gal[0]
            else:
                ind_mw = np.where((velocity > vmin_mw) & (velocity < vmax))
                ind_mw = ind_mw[0]
                ind_ivg = np.where((velocity > vmax_gal) & (velocity < vmin_mw))
                ind_ivg = ind_ivg[0]
                ind_gal = np.where((velocity > vmin)&  (velocity < vmax_gal))
                ind_gal = ind_gal[0]



            #ax[l+1].plot(velocity, spec, 'k')
            ax[l+1].fill_between(velocity[ind_mw], f[ind_mw], facecolor = mw_color, alpha = 0.3)
            ax[l+1].plot(velocity[ind_mw], f[ind_mw], mw_color)
            ax[l+1].fill_between(velocity[ind_ivg], f[ind_ivg], facecolor = hvc_color, alpha = 0.3)
            ax[l+1].plot(velocity[ind_ivg], f[ind_ivg], hvc_color)
            ax[l+1].fill_between(velocity[ind_gal], f[ind_gal],facecolor = gal_color, alpha = 0.3)
            ax[l+1].plot(velocity[ind_gal], f[ind_gal], gal_color)
            ax[l+1].plot(velocity, e, color ='gray')

            if vsys>0:
                dif_mw = np.abs(vmax_mw - velocity)
                imw = np.argmin(dif_mw)
                dif_ivg = np.abs(vmin_gal - velocity)
                iivg = np.argmin(dif_ivg)
                dif_gal = np.abs(vmax_gal - velocity)
                igal = np.argmin(dif_gal)
            else:
                dif_mw = np.abs(vmin_mw-velocity)
                imw = np.argmin(dif_mw)
                dif_ivg = np.abs(vmax_gal-velocity)
                iivg = np.argmin(dif_ivg)
                dif_gal = np.abs(vmin_gal -velocity)
                igal = np.argmin(dif_gal)

                this_window = np.where((velocity > vmin) &  (velocity < vmax))
            ymax = np.nanmax(f[this_window[0]])*1.5

            ax[l+1].axvline(x = vmax_mw, ymax = f[imw]/ymax,color = 'k', linestyle = '--')
            ax[l+1].axvline(x = vmin_gal, ymax =f[iivg]/ymax,color = 'k', linestyle = '--')
            ax[l+1].axvline(x = vmax_gal,ymax = f[igal]/ymax, color = 'k', linestyle = '--')

            ax[l+1].set_xlim(left=vmin, right=vmax)
            ax[l+1].set_ylabel("Flux", fontsize = 16)

            ax[l+1].set_ylim(bottom=0., top=ymax)
            ax[l+1].text(vmin+10., 0.85*ymax, labels[l] , fontsize=14)
            if len(vel_comps)>0:
                if len(vel_comps) == 3:
                    colors = ['magenta', 'blue', 'darkorange']
                    xoffsets = [-120 ,-20 , 10]
                    yoffsets = [1., 1., 1.]
                else:
                    colors = ['magenta', 'blue', 'blue', 'darkorange']
                    xoffsets = [-50, -20, 10, 0]
                    yoffsets = [1., 1., 1., 1.]
                ylims = ax[l+1].get_ylim()
                for i in range(len(vel_comps)):
                    ax[l+1].axvline(x=vel_comps[i], ymin = 0.85, ymax = 1, linestyle = '--', color = colors[i])
                    ax[l+1].text(vel_comps[i] + xoffsets[i], ylims[1]*0.75*yoffsets[i], '{:3.0f}'.format(np.round(vel_comps[i])) , fontsize = 10, color = colors[i])
            if l < len(lines)-1:
                ax[l+1].set_xticklabels([])
            else:
                ax[l+1].set_xticks(np.arange(vmin, vmax, 100))
                ax[l+1].set_xticklabels((np.arange(vmin, vmax, 100)))
                fig.subplots_adjust(hspace = 0, wspace=0)
        else:
            ax[l+1].set_xlim(left=vmin, right=vmax)
            ax[l+1].set_ylabel("Flux", fontsize = 16)
            ax[l+1].set_ylim(bottom=0., top=ymax)
            ax[l+1].text(vmin+10., 0.85*ymax, labels[l] , fontsize=14)
            if l < len(lines)-1:
                ax[l+1].set_xticklabels([])
            else:
                ax[l+1].set_xticks(np.arange(vmin, vmax, 100))
                ax[l+1].set_xticklabels((np.arange(vmin, vmax, 100)))
                fig.subplots_adjust(hspace = 0, wspace=0)

    ax[len(lines)].set_xlabel('LSR Velocity [km/s]', fontsize = 16)
    fig.subplots_adjust(left = 0.17, right = 0.95, bottom = 0.05, top = 0.95, hspace=0, wspace = 0)
    #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    #plt.tight_layout()

    plt.savefig(plotname, format = 'pdf', dpi = 1000)
    plt.clf()
    plt.close()

def plot_fuse_spectra_blair(target, line = None):

    files = glob.glob("/astro/dust_kg/jduval/METAL/data/FUSE/{}/*.fits".format(target))
    for file in files:
        root = file.split('/')
        root = root[len(root)-1]
        s = fits.open(file)
        s = s[1].data

        if line != None:
            vel = (s['WAVE']- line)/line*3.e5
            xx = vel
        else:
            xx = s['WAVE']
        plt.plot(xx, s['FLUX'], label = root)
    plt.legend(fontsize = 13)
    if line !=None:
        xtitle = 'Velocity'
    else:
        xtitle = 'Wavelength'

    plt.xlabel(xtitle, fontsize = 18)
    plt.ylabel("Flux", fontsize= 18)
    if line !=None:
        plt.xlim([-100,500])
    plt.show()


def plot_fuse_spectra_mast(all_file, line = None):

    s = fits.open(all_file)
    for ext in range(1, len(s)):

        if line != None:
            vel = (s[ext].data['WAVE']- line)/line*3.e5
            xx = vel
        else:
            xx = s[ext].data['WAVE']

        flux = s[ext].data['FLUX']
        flux = flux/np.median(flux)
        plt.plot(xx, flux, label = s[ext].header['EXTNAME'])
    plt.legend(fontsize = 12)
    plt.show()

def spectrum_fits_to_ascii(input):

    spec = fits.open(input)
    spec = spec[1].data

    if '.gz' in input:
        new_file = input.replace(".fits.gz", ".dat")
    else:
        new_file = input.replace(".fits", ".dat")

    if 'WAVELENGTH' in spec.columns.names:
        w =  spec['WAVELENGTH'].flatten()
    else:
        w = spec['WAVE'].flatten()
    f = spec['FLUX'].flatten()
    if 'ERROR' in spec.columns.names:
        e = spec['ERROR'].flatten()
    elif 'ERR' in spec.columns.names:
        e = spec['ERR'].flatten()
    else:
        e = np.zeros_like(w)

    ind_sort = np.argsort(w)
    w = w[ind_sort]
    f = f[ind_sort]
    e = e[ind_sort]

    t = Table()
    t['WAVELENGTH'] =w
    t['FLUX'] = f

    if 'ERROR' in spec.columns.names or 'ERR' in spec.columns.names:
        t['ERROR'] = e



    ascii.write(t, new_file, overwrite=True)

    return(new_file)



def helio_to_lsr(vhelio, ra, dec):


    #20.0km/s, 18:03:50.24, +30:00:16.8
    #ra0 =270.9593333333333
    #dec0 = 30.004666666666665
    #vlsr = vhelio + 19.7*(np.cos((ra - ra0)*math.pi/180.)*np.cos(dec0/180.*math.pi)*np.cos(dec/180.*math.pi) + np.sin(ra0/180.*math.pi)*np.sin(dec/180.*math.pi))



    dtor = math.pi/180.
    solarmotion_ra = np.double(((18.+.3/60.+50.29/3.6e3)*15.)*dtor)
    solarmotion_dec = np.double((30.+0./6e1+16.8/3.6e3)*dtor)
    solarmotion_mag = np.double(20.0)


    racalc = ra*dtor
    deccalc = dec*dtor

    deldec2 = (deccalc-solarmotion_dec)/2.0
    delra2 =  (racalc-solarmotion_ra)/2.0
    sindis = np.sqrt( np.sin(deldec2)*np.sin(deldec2) + np.cos(solarmotion_dec)*np.cos(deccalc)*np.sin(delra2)*np.sin(delra2) )
    theta = 2.0*np.arcsin(sindis)

    vlsr = vhelio+solarmotion_mag*np.cos(theta)


    return(vlsr)

def annotate_lines(line_wavelengths, line_labels, w, f, ax, offsets = None):

    valid = np.where((f > 0.) & (f>0.7*np.median(f)))
    valid = valid[0]

    coefs = np.polyfit(w[valid], f[valid], 3)
    cont = np.poly1d(coefs)(w)

    for l in range(len(line_wavelengths)):

        minindex = np.argmin(np.abs(line_wavelengths[l]-w))
        yline = 1.3*cont[minindex]
        ytext = 1.7*cont[minindex]

        offs = np.zeros(len(line_wavelengths))
        if offsets !=None:
            offs = offsets

        ax.annotate(line_labels[l], xy = (line_wavelengths[l], yline), xytext = (line_wavelengths[l]- offs[l], ytext),arrowprops=dict(arrowstyle='-',facecolor='black'),horizontalalignment='center')


def find_components(w, f, line, window_file, smooth = 1, outname = 'nhi_fitting_components', main_only = False, vsys = 262., outdir = './', lsr = False, ra = 0., dec= 0., spacing = [30., 30., 30., 30.], guess = [0., 60., 120., 300.], plot_smooth = 3):

    target = outname.split("_")[0]


    vel,cont, rms, fn, rmsn = cont_fit(w, f, line , window_file, smooth = smooth, outname= outname + '_cont_fit', show=False, plt_vmin = vsys-500, plt_vmax = vsys + 500)


    windows = ascii.read(window_file)
    vmin = min(windows['col1'].data)
    vmax = max(windows['col2'].data)


    if lsr ==True:
        vel = helio_to_lsr(vel, ra, dec)
        vmin = helio_to_lsr(vmin, ra, dec)
        vmax = helio_to_lsr(vmax, ra, dec)

    fs = convolve(fn, Box1DKernel(smooth))
    ind= np.where((vel > vmin) & (vel < vmax))
    ind = ind[0]

    print("VEL BOUNDARIES ", vmin, vmax)

    fsp = convolve(fn, Box1DKernel(plot_smooth))
    fsp = fsp[ind]

    v = vel[ind]
    wave = w[ind]
    flux = fs[ind]

    minima_ind = argrelextrema(flux, np.less)

    min_vel= v[minima_ind]
    min_flux =flux[minima_ind]

    print("TEST minima ", min_vel)

    if main_only==False:
        ind_sort = np.argsort(1.- min_flux)
        ind_sort = ind_sort[::-1]
        min_vel = min_vel[ind_sort]
        min_flux = min_flux[ind_sort]

        valid = np.where(1.-min_flux > 3.*rmsn)

        min_vel = min_vel[valid[0]]
        min_flux = min_flux[valid[0]]


        min_vels = np.zeros(len(guess))
        min_fluxes = np.zeros(len(guess))

        print("MINIMA ", min_vel)

        for iv in range(len(guess)):
            this_v = guess[iv]
            index = np.where((min_vel > this_v-spacing[iv]/2.) & (min_vel < this_v + spacing[iv]/2.))
            index = index[0]
            if len(index) > 0:
                index_comp = np.argmin(min_flux[index])
                min_vels[iv] = min_vel[index[index_comp]]
                print("COMP ", this_v, min_vels[iv])
                #min_flux is in teh heavily smoothed spectrum, find flux in teh unsmoothed spectrum
                dist = np.abs(min_vels[iv] - v)
                ii = np.argmin(dist)
                min_fluxes[iv] = fsp[ii] #min_flux[index[index_comp]]


        valid = np.where(min_vels !=0.)
        min_flux = min_fluxes[valid]
        min_vel = min_vels[valid]




    else:
        #mw= np.where((min_vel > -100.) & (min_vel < 100.))
        #mw = mw[0]
        #gal = np.where((min_vel > vsys-100.) & (min_vel < vsys + 100.))
        #gal = gal[0]

        print("USING NEW METHOD")

        mw= np.where((v > -100.) & (v < 100.))
        mw = mw[0]
        gal = np.where((v > vsys-100.) & (v < vsys + 100.))
        gal = gal[0]

        #indgal = np.argmin(min_flux[gal])
        #indmw = np.argmin(min_flux[mw])
        indgal = np.argmin(flux[gal])
        indmw = np.argmin(flux[mw])
        print("DEBUG ", vsys -100., vsys + 100)
        print("MIN FLUX ", np.nanmin(flux), np.nanmin(fsp))

        #min_vel = np.array([min_vel[mw[indmw]], min_vel[gal[indgal]]])
        min_vel = np.array([v[mw[indmw]], v[gal[indgal]]])
        min_ind = np.zeros(2, dtype = 'uint32')
        #for iv in range(len(min_vel)):
        #    dist = np.abs(min_vel[iv] - v)
        #    ii = np.argmin(dist)
        #    min_ind[iv] = ii

        for iv in range(len(min_vel)):
            dist = np.abs(min_vel[iv] - v)
            ii = np.argmin(dist)
            min_ind[iv] = ii

        #min_flux = np.array([np.min(min_flux[mw]), np.min(min_flux[gal])])
        min_flux = np.array([fsp[min_ind[0]], fsp[min_ind[1]]])



    print("COMPONENTS ", min_vel, min_flux)
    plt.clf()
    plt.close()
    fig = plt.figure(figsize = (10,9))
    plt.plot(v,fsp , 'k-', min_vel, min_flux, 'ro', v, flux, 'b-')
    for i in range(len(min_vel)):
        plt.text(min_vel[i] -5., min_flux[i]-0.05, '{:4.1f}'.format(min_vel[i]) + ' km/s', fontsize = 14, color = 'red')
    if lsr ==True:
        xtitle = 'LSR Velocity (km/s)'
    else:
        xtitle = 'Heliocentric Velocity (km s' + r'${-1}$)'
    plt.xlabel(xtitle, fontsize = 18)
    plt.ylabel("Normalized Spectrum", fontsize = 18)
    plt.text(200, 1.3, str(line) + ' A', fontsize = 18)
    plt.text(200, 1.4, target, fontsize = 18)
    plt.ylim([-0.1, 1.5])
    fig.tight_layout()
    plt.savefig(outdir + outname + ".pdf", format = "pdf", dpi = 1000)
    plt.clf()
    plt.close()

    return(vel, fn, rmsn, min_vel, min_flux)


def cont_fit(w, f, line, window_file,  degree = 3, smooth = 1, outname = 'nhi_fitting_cont_fit', plt_vmin = -200., plt_vmax = 600., outdir = './', spline=False, show=True):


    target = outname.split("_")[0]

    vel = (w-line)/line*3.e5
    if smooth > 1:
        fs = convolve(f, Box1DKernel(smooth))
    else:
        fs = f

    if (os.path.isfile(window_file)==False):
        print("CREATE THE WINDOW FILE ", window_file)
        plt.clf()
        plt.close()
        plt.plot(vel, fs, 'k-')
        plt.xlim([plt_vmin, plt_vmax])
        plt.ylim([0., 5.*np.nanmedian(fs[np.where((np.isnan(fs)==False) & (np.isinf(fs)==False))])])
        plt.show()
        plt.clf()
        plt.close()

    cont_win =  ascii.read(window_file)
    vmin = cont_win['col1'].data
    vmax = cont_win['col2'].data

    mask = np.isnan(f)==True

    cont_index = get_cont_index(vmin, vmax, vel, mask = mask)
    spec_to_fit = fs[cont_index]
    vel_to_fit  = vel[cont_index]
    w_to_fit = w[cont_index]

    if spline==False:
        print("USING LEGENDRE")
        coefs = np.polynomial.legendre.legfit(vel_to_fit, spec_to_fit, degree)
        cont = np.polynomial.legendre.legval(vel, coefs)
    else:
        print("USING SPLINE")
        spl= UnivariateSpline(vel_to_fit, spec_to_fit, k=5,check_finite=True)
        cont = spl(vel)



    rms = np.nanstd((spec_to_fit - cont[cont_index]))
    rmsn = np.nanstd((spec_to_fit/cont[cont_index]) - 1.)

    fn = f/cont

    plt.clf()
    plt.close()
    fig = plt.figure(figsize = (15, 8))
    plt.plot(vel, fs, 'k-', label = 'Spectrum')
    plt.plot(vel_to_fit, spec_to_fit, 'b.', label = 'Range to fit', alpha = 0.5)
    plt.plot(vel, cont, 'r-', label = 'Continuum')
    plt.xlim([plt_vmin, plt_vmax])
    plt.ylim([0, 2.*np.nanmedian(spec_to_fit)])
    plt.xlabel("Heliocentric Velocity (km s" + r'$^{-1}$)', fontsize = 18)
    plt.ylabel("Flux (" + r'erg s$^{-1}$' + ' ' + r'cm$^{-2}$' + ' ' + r'A$^{-1}$' + ')' , fontsize = 18)
    plt.text(0, np.nanmedian(spec_to_fit)*1.6, str(line) + ' A', fontsize = 18)
    plt.text(0, np.nanmedian(spec_to_fit)*1.8, target, fontsize = 18)
    plt.legend(fontsize = 16)
    fig.tight_layout()
    fig =plt.gcf()
    fig.savefig(outdir + outname + '.pdf', format = 'pdf', dpi = 1000)
    if show:
        plt.show()
    plt.clf()
    plt.close()

    return(vel, cont, rms, fn, rmsn)



def get_cont_index(vmin, vmax, vel, mask = []):

    nwin = len(vmin)


    for i in range(nwin):
        if len(mask)>0:
            this_ind = np.where((vel > vmin[i]) & (vel < vmax[i])& (mask==False))
        else:
            this_ind = np.where((vel > vmin[i]) & (vel < vmax[i]))
        this_ind = this_ind[0]
        if i == 0:
            cont_index = this_ind
        else:
            cont_index = np.concatenate((cont_index, this_ind))

    return(cont_index)



def get_fit_index(w, wmin, wmax, tau, tau_max = 3, tau_min = 0.3):

    nwin = len(wmin)


    for i in range(nwin):
        this_ind = np.where((w > wmin[i]) & (w < wmax[i]) & (tau <= tau_max) & (tau >= tau_min))
        this_ind = this_ind[0]
        if i == 0:
            fit_index = this_ind
        else:
            fit_index = np.concatenate((fit_index, this_ind))

    return(fit_index)



def nhi_fit_proc(ra, dec, spec_file, type_ascii = False,  target = '', smooth_sii = 3, smooth_hi = 9,outdir = './', tau_max = 3, show=False, tau_min  = 0.3, sampling = 0.01, vsys = 262., galaxy = 'LMC', use_vsys=False):
    """
    Main function to fit the HI lorentzian profile and measure N(HI) for the MW and the galaxy observed
    INPUTS:
    ** spec_file: path to spectrum file (fits or ascii, if ascii, set type_ascii=True)
    ** target: name of target (for filename purposes)
    ** vsys: systemic velocity of galaxy. The code will measure the velocity from S II lines if use_vsys is False (default). Otherwise, it will use the input systemic velocity
    ** sampling: sampling of log N(HI) grid. Default is 0.01 dex
    ** tau_min, tau_max; min/max tau included in fit (if tau is too high, there's not flux, if it's too small, measurements arent' meaningful
    """

    if type_ascii==False:
        specf = spectrum_fits_to_ascii(spec_file)
    else:
        specf = spec_file

    spec = ascii.read(specf)

    ws = spec['WAVELENGTH'].data
    fs = spec['FLUX'].data
    es  = spec['ERROR'].data

    #mask the nans
    
    nanarray = np.isnan(fs) #mask nans in lfux
    notnan = ~ nanarray

    f = fs[notnan]
    w = ws[notnan]
    e = es[notnan]
    
    l0 = np.double(1215.67) #Lyman alpha
    lsii = 1250.578

    if use_vsys==False:
    #determinethe continuum near the sii line, normalize the spectrum, get the velocity components from the S II 1250 A line
        vel_sii, fn_sii, rmsn_sii, vc_sii, fc_sii = find_components(w, f, lsii, "HI-FIT/" + target + '_sii_windows.dat',  smooth = smooth_sii, outname =target+ '_sii_components', main_only=True, vsys = vsys, outdir  = outdir)
    else:
        vc_sii = [0, vsys]
    #the MW component is the one at low evelocity
    ind_sort = np.argsort(vc_sii)

    if vsys > 50:
        v_mw = vc_sii[ind_sort[0]]
        v_gal = vc_sii[ind_sort[1]]
    else:
        v_mw = vc_sii[ind_sort[1]]
        v_gal = vc_sii[ind_sort[0]]
        print("VELOCITIES ", v_mw, v_gal)
    l0_mw = l0*(1. + v_mw/3.e5)
    l0_gal = l0*(1. + v_gal/3.e5)

    #fit teh continuum to lyman-alpha
    vel_hi,cont_hi, rms_hi, fn_hi, rmsn_hi = cont_fit(w, f, l0 , "HI-FIT/" + target + '_lyman_alpha_windows.dat', smooth = 5, plt_vmin = -20000., plt_vmax= 28000., degree=1, outname = target +  '_hi_cont_fit',outdir   = outdir, show=show)

    #find the best fit to the continuum in fit windows

    grid_mw = np.arange(18.5, 22.5, sampling)
    grid_gal = np.arange(18.5, 22.5, sampling)

    if smooth_hi > 1:
        wmin = np.min(w)
        wmax = np.max(w)
        disp = np.median(np.abs(w-np.roll(w,1)))

        rdisp = disp*smooth_hi
        rw = np.arange(wmin + 1, wmax-1, rdisp)
        fs, es = spectres(rw, w, f, e)
        conts, ecs = spectres(rw, w, cont_hi, np.zeros_like(cont_hi))

    else:
        rw = w
        fs = f
        conts=cont_hi
        es= e

    #Get the rms between rebinned spetrum and continuum
    #If I forward model to estimate chi2, I want error on measurement
    vel_hel = 3.e5*(rw-l0)/l0
    cont_win =  ascii.read("HI-FIT/" + target + '_lyman_alpha_windows.dat')
    vmin = cont_win['col1'].data
    vmax = cont_win['col2'].data

    cont_std_index = get_cont_index(vmin, vmax, vel_hel)

    #the error array does not include noise due to weak stellar lines. We want to account for this, so measure teh "stellar line noise" empirically from the standard deviation of teh continuum and scale the error array accordingly. This wil also include fixed pattern noise.
    rms = np.nanstd((fs[cont_std_index] - conts[cont_std_index]))
    scaling_err = rms/np.median(es[cont_std_index])
    print("SCALING ERR ", scaling_err)
    es = es*scaling_err
    #es = np.zeros_like(es) + rms
    #print("RMS ", rms)

    #plt.clf()
    #plt.close()
    #plt.plot(fs[cont_std_index], np.roll(fs[cont_std_index],1), 'k.')
    #plt.show()
    #plt.clf()
    #plt.close()

    #best_hi_mw, mx_sigma_mw, p50_mw, c68_unc_mw, best_hi_gal, mx_sigma_gal, p50_gal, c68_unc_gal, fcorr, chi2 = fit_nhi(rw,fs,es,conts,cont_std_index, l0_mw, l0_gal, target + '_lyman_alpha_fit_windows.dat',grid_mw, grid_gal, smooth = smooth_hi, outname = target + '_hi_fit', outdir = outdir, tau_max = tau_max, tau_min = tau_min, galaxy = galaxy)
    nhi_mw_21cm, l_nhi_mw, best_hi_mw, best_fit_sigmalow_mw, best_fit_sigmahigh_mw, p50_mw, percentiles_sigmalow_mw,percentiles_sigmahigh_mw, l_nhi_gal, best_hi_gal, best_fit_sigmalow_gal, best_fit_sigmahigh_gal, p50_gal, percentiles_sigmalow_gal,percentiles_sigmahigh_gal, flux, chi2, wavelength, best_fit_model, tau = fit_nhi(ra, dec, rw,fs,es,conts,cont_std_index, l0_mw, l0_gal, "HI-FIT/" + target + '_lyman_alpha_fit_windows.dat',grid_mw, grid_gal, smooth = smooth_hi, outname = target + '_hi_fit2022', outdir = outdir, tau_max = tau_max, tau_min = tau_min, galaxy = galaxy)


    t = Table()
    #t['wavelength'] = w
    #t['flux'] = f
    t['rebinned wavelength'] = rw
    t['rebinned flux'] = flux
    t['model'] = best_fit_model
    t['tau'] = tau

    ascii.write(t, 'hi_model_{}.dat'.format(target), overwrite=True)

    return(nhi_mw_21cm, l_nhi_mw, best_hi_mw, best_fit_sigmalow_mw, best_fit_sigmahigh_mw, p50_mw, percentiles_sigmalow_mw,percentiles_sigmahigh_mw, l_nhi_gal, best_hi_gal, best_fit_sigmalow_gal, best_fit_sigmahigh_gal, p50_gal, percentiles_sigmalow_gal,percentiles_sigmahigh_gal, flux, chi2, v_mw, v_gal, wavelength, best_fit_model, tau)


def aod_proc(spec_file, line,flambda, type_ascii = False, target = '', smooth = 5, outdir = './', element = 'O I', vmin_line= 200., vmax_line = 330, show=True, spline=False, vsys = 262., cont_adjust=False,plot_sii=False):

    #Read the spectrum

    if type_ascii==False:
        specf = spectrum_fits_to_ascii(spec_file)
    else:
        specf = spec_file

    spec = ascii.read(specf)

    w = spec['WAVELENGTH'].data
    f = spec['FLUX'].data
    e  = spec['ERROR'].data

    #First identify velocity range from S II 1250
    line_sii =1250.578

    line_sii_key = '{:7.3f}'.format(line_sii)

    if plot_sii==True:
        vhelio_sii,cont_sii, rms_sii, fn_sii, rmsn_sii = cont_fit(w, f, line_sii ,  target + '_sii_windows.dat', smooth =13 , plt_vmin = vsys-800, plt_vmax= vsys + 800, degree=3, outname = target +  '_sii_cont_fit',outdir   = outdir, show = show)

        fns_sii = convolve(fn_sii, Box1DKernel(13))
        en_sii= e/cont_sii
        ens_sii = en_sii/np.sqrt(13)

    #plt.clf()
    #plt.close()
    #plt.plot(vhelio_sii, fn_sii, 'k', vhelio_sii, fns_sii, 'r')
    #plt.plot(vhelio_sii[signal_sii], fn_sii[signal_sii], 'b.')
    #plt.xlim([0,500])
    #plt.ylim([0,2])
    #plt.show()




    #fit the continuum

    line_key = '{:7.3f}'.format(line)

    tw = Table()


    if cont_adjust==False:
        if vsys>0:
            tw['col1'] = np.array([vmin_line-500, vmax_line])#np.array([-300., vmax_line])
            tw['col2'] = np.array([vmin_line, vmax_line + 500])#np.array([vmin_line, 700])

        else:
            tw['col1'] = np.array([vsys-300., vmax_line, 100.])
            tw['col2'] = np.array([vsys - 100., -100., 180.])

        ascii.write(tw, target + '_auto_cont_fit_' + line_key + '_windows.dat', overwrite=True)

    vhelio,cont, rms, fn, rmsn = cont_fit(w, f, line , target + '_auto_cont_fit_' + line_key + '_windows.dat', smooth = smooth, plt_vmin = vsys-800., plt_vmax= vsys + 800., degree=9, outname = target +  '_' + line_key + '_cont_fit',outdir   = outdir, show = show, spline=spline)

    en= e/cont

    if smooth > 1:
        fn = convolve(fn, Box1DKernel(smooth))
        en = en/np.sqrt(smooth)


    aod = np.log(1./fn)

    if plot_sii==True:
        vrange_sii = np.where((vhelio_sii>vmin_line) &  (vhelio_sii < vmax_line))
        vrange_sii = vrange_sii[0]
    vrange = np.where((vhelio>vmin_line) &  (vhelio < vmax_line) & (np.isnan(aod)==False))
    vrange = vrange[0]

    nav = aod*10.**(-1.*flambda)*10.**(14.576)

    delta = vmax_line-vmin_line
    noisem = np.where((vhelio>vmin_line-delta) & (vhelio < vmin_line) & (np.isnan(nav)==False) )
    noisep = np.where((vhelio>vmax_line) & (vhelio < vmax_line+delta) & (np.isnan(nav)==False) )
    noisem = noisem[0]
    noisep = noisep[0]

    na = np.trapz(nav[vrange], vhelio[vrange])
    noise  = 0.5*(np.std(nav[noisem]) +  np.std(nav[noisep]))
    noise_na = np.sqrt(len(vrange))*noise*np.median(np.abs(np.roll(vhelio[vrange],1)-vhelio[vrange]))
    #noise_na = 0.5*(np.trapz(nav[noisem], vhelio[noisem]) + np.trapz(nav[noisep], vhelio[noisep])
    logna =np.log10(na)
    noise_logna = noise_na/na/np.log(10.)

    EW = np.trapz(1.-fn[vrange], w[vrange])
    SN = np.double(0.5*(np.nanmedian(fn[noisem]) + np.nanmedian(fn[noisep])))/np.double((0.5*(np.nanstd(fn[noisem]) + np.nanstd(fn[noisep]))))
    #This is a proxy and assumes COS resolution. Need errors from Jenkins reference
    err_EW = line/20000./SN

    mn = np.nanmin(nav[vrange])
    mx  = np.nanmax(nav[vrange])

    print("COLUMN ", logna, noise_logna)

    plt.clf()
    plt.close()
    if plot_sii==True:
        fig, ax = plt.subplots(3,sharex=True,figsize = (5,7))
    else:
        fig, ax = plt.subplots(2,sharex=True,figsize = (5,5))

    if vsys>0:
        xmin = vsys-800
        xmax= vsys + 800
    else:
        xmin = vsys-300
        xmax = 200

    if plot_sii==True:
        ax[0].plot(vhelio_sii, fn_sii, 'k')
        ax[0].set_ylim(bottom = 0, top = 1.5)
        ax[0].set_xlim(left = xmin, right = xmax)
        ax[0].set_title(target, fontsize = 20)
        ax[0].set_xticklabels([])
        ax[0].text(50, 1.3, 'S II ' + line_sii_key, fontsize = 18)
        ax[0].fill_between(vhelio_sii[vrange_sii], fn_sii[vrange_sii], facecolor = 'r', alpha = 0.3, linestyle = '--')
        ax[0].set_ylabel("Normalized flux", fontsize = 18)
        istart = 1
    else:
        istart = 0

    dumx= np.arange(vsys-800,vsys + 800,1)
    dumy = np.zeros(len(dumx)) +1.
    ax[istart].plot(vhelio, fn, 'k')
    ax[istart].plot(dumx, dumy, 'b--')
    ax[istart].fill_between(vhelio[vrange], fn[vrange], facecolor = 'r', alpha = 0.3, linestyle = '--')
    ax[istart].set_xticklabels([])
    ax[istart].set_ylabel("Normalized flux", fontsize = 18)
    ax[istart].set_ylim(bottom = 0, top = 1.5)
    ax[istart].set_xlim(left =xmin, right = xmax)

    dumx= np.arange(xmin,xmax+100,1)
    dumy = np.zeros(len(dumx))
    ax[istart+1].plot(vhelio, nav, 'k')
    ax[istart+1].plot(dumx, dumy, 'k--')
    #ax[1].set_yscale('log')
    ax[istart+1].set_xlabel("Heliocentric Velocity (km " + r's$^{-1}$' + ')', fontsize = 18)
    ax[istart+1].set_ylabel("Column density (" + r'cm$^{-2}$' + ')', fontsize = 18)

    #plt.plot(vhelio, ens, 'r')
    #ax[1].set_ylim(bottom = mn*0.1 ,  top  =  10.*mx)
    ax[istart+1].set_ylim(bottom = mn ,  top  =  1.3*mx)
    ax[istart+1].fill_between(vhelio[vrange],np.zeros(len(vrange)) , nav[vrange], facecolor = 'r', alpha = 0.3, linestyle = '--')

    ax[istart+1].text(vsys + 100, mx*1.1, element + ' ' + '{:6.2f}'.format(line), fontsize = 20)
    ax[istart+1].text(vsys + 100, mx *0.9, "log N = " + '{:4.2f}'.format(np.log10(na)) + " " + r'$\pm$' + " " + '{:3.2f}'.format(noise_logna) + " " +  r'cm$^{-2}$', fontsize = 20)
    ax[istart+1].set_xticks(np.arange(xmin, xmax, 50))
    ax[istart+1].set_xticklabels(np.arange(xmin, xmax, 50))
    ax[istart+1].set_xlim(left =xmin, right = xmax)

    fig.subplots_adjust(hspace = 0, bottom = 0.1, left = 0.1, top = 0.95, right = 0.95)
    fig = plt.gcf()
    fig.savefig(outdir + "plot_aod_" + target + '_' +  element.replace(' ', '') + "_" + line_key + '.pdf', format = 'pdf', dpi = 1000)
    if show:
        plt.show()
    plt.clf()
    plt.close()

    spec_tab = Table()
    incl = np.where((vhelio >= xmin-300) & (vhelio <= xmax + 200))
    incl = incl[0]
    spec_tab['wavelenth'] = w[incl]
    spec_tab['vhelio']  = vhelio[incl]
    spec_tab['fn'] = fn[incl]
    spec_tab['AOD'] = aod[incl]
    spec_tab['Nav'] = nav[incl]
    ascii.write(spec_tab, 'AOD_results_'+  '{:6.2f}'.format(line) + '_'  + target + '.dat')

    return(logna, noise_logna, EW, err_EW)


def threshold_morph_spectrum(spectrum, emission_threshold_spectrum, structure_size, value = 0.):
    #calculates the mask such that mask = True if emission_threshold_spetrum > threshold

    test = np.where(emission_threshold_spectrum >2)
    test = test[0]
    mask = np.zeros_like(spectrum)

    if len(test)>0:
        structure = np.ones(structure_size)
        mask[emission_threshold_spectrum>1]=1
        e = binary_erosion(mask, structure = structure)
        d = binary_dilation(e, structure = structure)
    else:
        d= mask
    output = np.zeros_like(spectrum) + value
    output[d==True] = spectrum[d==True]

    #plt.plot(emission_threshold_spectrum, 'k')
    #plt.plot(emission_threshold_spectrum*mask, 'r.')
    #plt.plot(emission_threshold_spectrum*d, 'b.')
    #plt.show()
    #plt.clf()
    #plt.close()

    return(output, d)





def fit_nhi(ra, dec, rw,fs,es,conts,cont_std_index,l0_mw, l0_gal, fit_window_file,grid_mw, grid_gal, smooth = 5, outname = 'nhi_fitting', outdir = './', tau_max = 3., tau_min = 0.3, show=False, galaxy  = 'LMC'):

        """
        Fits a lorentzian profile to the Ly-a 1216 line to determine HI column densities in the galaxy observed and MW
        inputs are:
        ** rw: wavelength array
        ** fs: flux array
        ** es: error array
        ** cont_std_index: index relative to rw, fs, es of where to estimate the continuum
        ** l0_mw: wavlength of the center Ly-a MW absorption
        ** l0_gas: wavelength of the center Ly-a galaxy absorption
        ** fit_window_file: path to text file with the wavelength intervals to be used for fitting the profile
        ** grid_mw: 1D grid of N(HI) values for the MW component
        ** grid_gal: 1D grid of N(HI) values for the gal component

        OUTPUTS:
        best_hi_mw, mx_sigma_mw, p50_mw, c68_unc_mw, best_hi_gal, mx_sigma_gal, p50_gal, c68_unc_gal, fs, chi2
        ** best_hi_mw, gal: best fit N(HI) for MW, gal
        ** best_fit_sigma{low, high}_{mw, gal} low/high uncertainties computed from the 68% confidence level (chi2_min+2.3) around the best fit
        ** p50_mw, gal: N(HI) 50th percentile value for MW, gal
        ** percentiles_sigma{low, high}_{mw,gal}: N(HI) uncertainties computed as p50-16% and 84%-p50 (68% confidence interval centered on p50)
        ** fs: input spectrum
        ** chi2: chi2!
        """

        l0 = np.double(1215.67)
        vmw = 3.e5*(l0_mw-l0)/l0
        vgal = 3.e5*(l0_gal-l0)/l0

        target = outname.split('_')[0]

        print("TAU_MAX ", tau_max)
        print("TAU_MIN ", tau_min)

        sampling_mw = np.median(np.abs(grid_mw-np.roll(grid_mw,1)))
        sampling_gal = np.median(np.abs(grid_gal-np.roll(grid_gal,1)))

        nmw = len(grid_mw)
        ngal = len(grid_gal)

        ln_prob = np.zeros([nmw, ngal], dtype = 'float32')
        chi2 = np.zeros([nmw, ngal], dtype = 'float32')

        if os.path.isfile(fit_window_file)==False:
            print("CREATE WINDOWS FOR CHI2 ESTIMATION")
            plt.plot(rw, fs, 'k')
            #plt.plot(rw, es*np.exp(3.), 'b')
            plt.xlim([1130, 1320])
            good = np.where((np.isnan(fs)==False) & (np.isinf(fs)==False))
            plt.ylim([0, 2.5*np.nanmedian(fs[good])])
            plt.show()

        t = ascii.read(fit_window_file)
        wmin = t['col1'].data
        wmax = t['col2'].data

        #implemented the N(HI) depdendent shift in teh scattering cross section from Lee+2003
        sigma_mw = 4.26e-20/(6.04e-10 + (np.double(rw)-l0_mw)**2)*(1.-1.792*(np.double(rw)-l0_mw)/l0_mw)
        sigma_gal = 4.26e-20/(6.04e-10 + (np.double(rw)-l0_gal)**2)*(1.-1.792*(np.double(rw)-l0_gal)/l0_gal)

        #fit_index= get_cont_index(wmin, wmax,rw)

        for i in range(nmw):
            for j in range(ngal):

                hi_mw = grid_mw[i]
                hi_gal = grid_gal[j]

                tau_mw = 10.**(hi_mw)*sigma_mw
                tau_gal= 10.**(hi_gal)*sigma_gal
                correction = np.exp(tau_mw)*np.exp(tau_gal)
                tau = np.log(correction)

                corr_f = fs*correction
                corr_e = es*correction

                #Fixed the windows, does nto depend on tau anymore
                fit_index = get_fit_index(rw, wmin, wmax, tau, tau_max = tau_max, tau_min = tau_min)

                if len(fit_index)>0:

                    #print("OK ", grid_mw[i], grid_gal[j])

                    spec_to_fit = fs[fit_index]
                    err_to_fit = es[fit_index]
                    tau_to_fit = tau[fit_index]
                    w_to_fit = rw[fit_index]

                    cont_to_fit = conts[fit_index]

                    #if grid_gal[j]>=20.:
                    #    plt.plot(w_to_fit, spec_to_fit, 'k.')
                    #    plt.plot(w_to_fit, cont_to_fit, 'r.')
                    #    this_model = cont_to_fit*np.exp(-tau_to_fit)
                    #    plt.plot(w_to_fit, this_model, 'g.')
                    #    plt.show()


                    #for COS, teh gap can be nan, so filter it out

                    valid = np.where((np.isnan(spec_to_fit) ==False) & (np.isinf(spec_to_fit)==False) & (err_to_fit > 0.))
                    valid = valid[0]
                    n = len(valid)

                    chi2[i,j] = np.double(np.nansum((spec_to_fit[valid] - cont_to_fit[valid]*np.exp(-tau_to_fit[valid]))**2/err_to_fit[valid]**2))
                    #print("DEBUG ", grid_mw[i], grid_gal[j], chi2[i,j])

                else:
                    #print("PB ", grid_mw[i], grid_gal[j])
                    chi2[i,j] = 100000000.

                ln_prob[i,j] = - chi2[i,j]/2.
        #the max is too high and turns into nan numerically, so subtract teh max, then normalize
        ln_prob = ln_prob-np.max(ln_prob)

        ln_prob = np.double(ln_prob) + np.log(1.) - np.log(np.sum(np.exp(np.double(ln_prob))))#- np.log(np.median(grid_mw-np.roll(grid_mw,1)))
        prob= np.exp(np.double(ln_prob))

        min_index = np.argmax(ln_prob)
        ij = np.unravel_index(min_index, ln_prob.shape)

        best_hi_mw = grid_mw[ij[0]]
        best_hi_gal = grid_gal[ij[1]]

        tau_mw = 10.**(best_hi_mw)*sigma_mw
        tau_gal= 10.**(best_hi_gal)*sigma_gal
        correction = np.exp(tau_mw)*np.exp(tau_gal)
        tau= np.log(correction)
        #fcorr = f*correction
        fcorrs = fs*correction
        cont_corr = conts*np.exp(-tau)
        bad = np.where(tau > 10)
        #fcorr[bad] = np.nan
        fcorrs[bad] = np.nan

        fit_index = get_fit_index(rw, wmin, wmax, tau, tau_max = tau_max,tau_min= tau_min)
        #fit_index = get_cont_index(wmin, wmax, w)
        spec_to_fit = fs[fit_index]
        w_to_fit = rw[fit_index]
        cont_to_fit = conts[fit_index]

        hdu = fits.PrimaryHDU(ln_prob)
        hdu2 = fits.ImageHDU(chi2 - np.nanmin(chi2))
        hdulist = fits.HDUList([hdu, hdu2])
        hdulist.writeto(outname + "_ln_prob.fits", overwrite=True)
        #bad = np.where((np.isnan(prob) ==True) | (np.isinf(prob)==True))
        #prob[bad] = 0.
        mx_pdf_mw = np.nanmax(prob, axis = 1)
        mx_pdf_gal = np.nanmax(prob, axis = 0)
        mx_pdf_mw = mx_pdf_mw/trapezoid(mx_pdf_mw, grid_mw)   #np.nansum(mx_pdf_mw)
        mx_pdf_gal = mx_pdf_gal/trapezoid(mx_pdf_gal, grid_gal)#np.nansum(mx_pdf_gal)

        int_pdf_mw = trapezoid(prob, grid_gal, axis = 1)#np.nansum(prob, axis = 1)
        int_pdf_gal = trapezoid(prob, grid_mw, axis = 0)#np.nansum(prob, axis = 0)
        int_pdf_mw = int_pdf_mw/trapezoid(int_pdf_mw, grid_mw)#np.nansum(int_pdf_mw)
        int_pdf_gal = int_pdf_gal/trapezoid(int_pdf_gal, grid_gal)#np.nansum(int_pdf_gal)

        #write the _PDFs
        pdf_table_mw = Table()
        pdf_table_mw['N(HI)'] = grid_mw
        pdf_table_mw['MX_PDF'] = mx_pdf_mw
        pdf_table_mw['INT_PDF'] = int_pdf_mw
        pdf_mw_file = outdir + outname + '_mw_pdf.dat'
        ascii.write(pdf_table_mw, pdf_mw_file, format = 'csv', overwrite=True)

        pdf_table_gal = Table()
        pdf_table_gal['N(HI)'] = grid_gal
        pdf_table_gal['MX_PDF'] = mx_pdf_gal
        pdf_table_gal['INT_PDF'] = int_pdf_gal

        pdf_gal_file = outdir + outname + '_gal_pdf.dat'
        ascii.write(pdf_table_gal, pdf_gal_file, format = 'csv', overwrite=True)


        #Now determine uncertaintiesin 2 ways.
        #First, get the contour of chi2 = chi2min + 2.3 (0.68 confidence interval, and just report the left and right distance from best fit in each direction)
        #second, compute the percentiles of the marginalized _PDFs

        #method 1: use the best fit, adn distance to the min/max (X,Y) in the chi2 contour lower than chi2min + 2.3

        chi2_min = np.nanmin(chi2)
        confidence_interval = np.where(chi2 <= chi2_min + 2.3)
        #confidence_interval2 = np.unravel_index(confidence_interval1[0], chi2.shape)
        nhmin_mw = grid_mw[np.min(confidence_interval[0])]
        nhmax_mw = grid_mw[np.max(confidence_interval[0])]

        nhmin_gal = grid_gal[np.min(confidence_interval[1])]
        nhmax_gal = grid_gal[np.max(confidence_interval[1])]

        best_fit_sigmalow_mw = best_hi_mw - nhmin_mw
        best_fit_sigmahigh_mw = nhmax_mw - best_hi_mw

        best_fit_sigmalow_gal = best_hi_gal - nhmin_gal
        best_fit_sigmahigh_gal = nhmax_gal - best_hi_gal

        #now method 2: use the percentiles of the marginalized PDFs.


        percentiles_mw, xpercentiles_mw = get_percentiles(grid_mw, int_pdf_mw, percentiles =[0.16, 0.32, 0.5, 0.68, 0.84] )
        percentiles_gal, xpercentiles_gal = get_percentiles(grid_gal, int_pdf_gal, percentiles =[0.16, 0.32, 0.5, 0.68, 0.84] )

        p50_mw = xpercentiles_mw[2]
        p50_gal = xpercentiles_gal[2]

        percentiles_sigmalow_mw = xpercentiles_mw[2] - xpercentiles_mw[0]
        percentiles_sigmahigh_mw = xpercentiles_mw[4] - xpercentiles_mw[2]

        percentiles_sigmalow_gal = xpercentiles_gal[2] - xpercentiles_gal[0]
        percentiles_sigmahigh_gal = xpercentiles_gal[4] - xpercentiles_gal[2]

        print("RESULT MW ", best_hi_mw, best_fit_sigmalow_mw, best_fit_sigmahigh_mw , p50_mw, percentiles_sigmalow_mw ,percentiles_sigmahigh_mw )
        print("RESULT " + galaxy, best_hi_gal, best_fit_sigmalow_gal, best_fit_sigmahigh_gal , p50_gal, percentiles_sigmalow_gal ,percentiles_sigmahigh_gal)

        #Now turn large unc into limits:
        l_nhi_mw = ''
        l_nhi_gal = ''
        if best_hi_mw == np.min(grid_mw):
            l_nhi_mw = '<'
            best_hi_mw = best_hi_mw + best_fit_sigmahigh_mw
        if best_hi_gal == np.min(grid_gal):
            l_nhi_gal = '<'
            best_hi_gal = best_hi_gal + best_fit_sigmahigh_gal



        nhi_mw_21cm = np.log10(get_mw_nhi21cm(ra, dec))

        plt.clf()
        plt.close()
        fig = plt.figure(figsize = (10,9))
        plt.plot(grid_mw, int_pdf_mw, 'k', label  = "MW")
        plt.plot(grid_gal, int_pdf_gal, 'r' , label = galaxy)
        plt.plot(np.zeros(11) + nhi_mw_21cm, np.arange(11)*np.max(int_pdf_gal)/10., '--', color = 'lightgray', label = 'MW 21 cm')
        plt.plot(np.zeros(11) + best_hi_mw, np.arange(11)*np.max(int_pdf_gal)/10., '--', color = 'black', label = 'MW best fit', alpha = 0.5)
        if l_nhi_mw == '<':
            plt.errorbar([best_hi_mw], np.max(int_pdf_gal), fmt = '.', xerr = 0.5, xuplims = [1], color = 'black', alpha= 0.5)

        if l_nhi_gal == '<':
            plt.errorbar([best_hi_gal], np.max(int_pdf_gal), fmt = '.', xerr = 0.5, xuplims = [1], color = 'red', alpha= 0.5)
        plt.plot(np.zeros(11) + best_hi_gal, np.arange(11)*np.max(int_pdf_gal)/10., '--', color = 'red', label = '{} best fit'.format(galaxy), alpha = 0.5)
        plt.xlabel("log N(HI) (cm" + r'$^{-2}$)', fontsize = 18)
        plt.ylabel("PDF", fontsize = 18)
        plt.legend(fontsize = 18)
        plt.text(min(grid_mw) + 0.1, max(int_pdf_mw)*0.5, target, fontsize = 18)
        #plt.yscale('log')
        fig.tight_layout()
        fig = plt.gcf()
        fig.savefig(outdir + outname + "_pdfs.pdf", format = "pdf", dpi = 1000)
        #plt.show()

        plt.clf()
        plt.close()

        #fig,ax = plt.subplots(figsize = (11,9))
        #im=ax.imshow(prob, vmin = np.max(prob)/10., vmax = np.max(prob), origin = 'lower', extent = [np.min(grid_gal), np.max(grid_gal), np.min(grid_mw), np.max(grid_mw)], cmap = 'gist_stern_r')
        #ax.contour(grid_gal, grid_mw, prob, levels = [0.6*np.max(prob)], colors  = ['magenta'])
        #ax.set_xlabel("LMC log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
        #ax.set_ylabel("MW log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
        #ax.text(min(grid_gal) + 0.1, min(grid_mw)+0.1, target, fontsize = 18)
        #cbaxes = inset_axes(ax, width="80%", height="5%", loc=9)
        #cbar = fig.colorbar(im,cax=cbaxes, orientation='horizontal')
        ##cbar =fig.colorbar(im,orientation='vertical', fraction  = 0.04, ax = ax)
        #cbar.set_label("p = exp(-" + r'$\chi^2/2)$', fontsize = 18)
        #fig.subplots_adjust(bottom = 0.1, top = 0.95, left = 0.1, right = 0.85)
        #plt.savefig(outdir + outname + '_prob.pdf', format = 'pdf', dpi = 1000)
        #plt.clf()
        #plt.close()

        fig = plt.figure(figsize = (11,11))
        grid = plt.GridSpec(5, 5, hspace=0., wspace=0.)
        main_ax = fig.add_subplot(grid[:-1,1:])
        y_hist = fig.add_subplot(grid[:-1,0], xticklabels = [], sharey=main_ax)
        x_hist = fig.add_subplot(grid[-1,1:], yticklabels = [], sharex=main_ax)

        # scatter points on the main axes
        im = main_ax.imshow(prob, vmin = 0., vmax = np.max(prob), origin = 'lower', extent = [np.min(grid_gal), np.max(grid_gal), np.min(grid_mw), np.max(grid_mw)], cmap = 'gist_stern_r')
        #im = main_ax.imshow(prob, vmin = np, vmax = np.nanmax(chi2), origin = 'lower', extent = [np.min(grid_gal), np.max(grid_gal), np.min(grid_mw), np.max(grid_mw)], cmap = 'gist_stern_r')
        main_ax.contour(grid_gal, grid_mw, prob, levels = [np.nanmax(prob)*np.exp(-2.3/2.)], colors  = ['orange'])
        main_ax.axhline(y = nhi_mw_21cm, linestyle = '--', color = 'lightgray')
        main_ax.axhline(y = best_hi_mw, linestyle = '--', color = 'red', alpha = 0.5)
        main_ax.axvline(x = best_hi_gal, linestyle = '--', color = 'red', alpha = 0.5)



        main_ax.text(min(grid_gal) + 0.1, min(grid_mw)+0.1, target, fontsize = 18)
        main_ax.set_xticklabels([])
        main_ax.set_yticklabels([])
        cbaxes = inset_axes(main_ax, width="80%", height="5%", loc=9)
        cbar = fig.colorbar(im,cax=cbaxes, orientation='horizontal')
        #cbar =fig.colorbar(im,orientation='horizontal', fraction  = 0.04, gap = -0.04, ax = main_ax)
        cbar.set_label("p ~ exp(-" + r'$\chi^2/2)$', fontsize = 18)

        #a zoom panel
        zax = plt.axes([0.65, 0.65, 0.2, 0.2])
        delta_mw = min([percentiles_sigmalow_mw/sampling_mw*5., 0.5/sampling_mw])
        delta_gal = min([percentiles_sigmalow_gal/sampling_gal*5.,0.5/sampling_gal])
        xmin_mw = int(max([ij[0]-delta_mw,0]))
        xmax_mw = int(min([ij[0] + delta_mw, nmw]))
        xmin_gal = int(max([ij[1]-delta_gal,0]))
        xmax_gal = int(min([ij[1] + delta_gal, ngal]))

        zax.imshow(prob[xmin_mw:xmax_mw, xmin_gal:xmax_gal], vmin = 0., vmax = np.nanmax(prob), origin = 'lower', extent = [np.min(grid_gal[xmin_gal:xmax_gal]), np.max(grid_gal[xmin_gal:xmax_gal]), np.min(grid_mw[xmin_mw:xmax_mw]), np.max(grid_mw[xmin_mw:xmax_mw])], cmap = 'gist_stern_r')
        zax.contour(grid_gal[xmin_gal:xmax_gal], grid_mw[xmin_mw:xmax_mw], prob[xmin_mw:xmax_mw, xmin_gal:xmax_gal], levels = [np.nanmax(prob)*np.exp(-2.3/2.)], colors  = ['orange'])

        # histogram on the attached axes
        y_hist.plot(int_pdf_mw,grid_mw, 'r')
        y_hist.plot(np.arange(11)/10.*np.max(int_pdf_mw), np.zeros(11) + nhi_mw_21cm, '--', color = 'lightgray')
        y_hist.plot(np.arange(11)/10.*np.max(int_pdf_mw), np.zeros(11) + best_hi_mw, '--', color = 'black', alpha = 0.5)
        if l_nhi_mw == '<':
            y_hist.errorbar([0.1], [best_hi_mw], yerr = 0.5, uplims =[1], color = 'black', alpha = 0.5)

        #y_hist.xaxis.tick_top()
        #y_hist.set_ylim([np.min(grid_mw), np.max(grid_mw)])
        y_hist.set_ylabel("MW log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
        y_hist.set_yticks(np.arange(19,22.5, 0.5))
        y_hist.set_yticklabels(np.arange(19,22.5, 0.5))
        y_hist.set_xlabel("PDF", fontsize = 18)
        y_hist.invert_xaxis()

        x_hist.plot(grid_gal,int_pdf_gal, 'r')
        x_hist.plot(np.zeros(11) + best_hi_gal,np.arange(11)/10.*np.max(int_pdf_gal),  '--', color = 'black', alpha = 0.5)
        if l_nhi_gal == '<':
            x_hist.errorbar([best_hi_gal], [np.max(int_pdf_gal)], xerr = 0.5, xuplims =[1], color = 'black', alpha = 0.5)
        x_hist.set_xlabel(galaxy + " log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
        x_hist.set_xticks(np.arange(19,22.5, 0.5))
        x_hist.set_xticklabels(np.arange(19,22.5, 0.5))
        x_hist.set_ylabel("PDF", fontsize = 18)
        x_hist.invert_yaxis()

        fig.subplots_adjust(bottom = 0.1, top = 0.95, left = 0.1, right = 0.95)
        plt.savefig(outdir + outname + '_prob.pdf', format = 'pdf', dpi = 1000)
        plt.clf()
        plt.close()

        good = np.where((np.isnan(fs) ==False) & (np.isinf(fs)==False))
        good = good[0]

        fig = plt.figure(figsize = (15, 8))
        plt.plot(rw, fs, 'k', label = 'Spectrum')
        plt.plot(rw, es, '-', color = 'gray', label = 'Error')
        plt.plot(rw[cont_std_index], fs[cont_std_index], '.', color = 'magenta', label = 'Continuum est.')
        plt.plot(w_to_fit, fs[fit_index], 'b.',  label = r'$\chi^2$ minimzation range')
        #plt.plot(w_to_fit, spec_to_fit, 'b.', label = r'$\chi^2$ minimzation range')
        plt.plot(rw, fcorrs, 'r', alpha = 0.3, label = 'Reconstructed spectrum (x ' + r'e$^{\tau}$' + ')')
        plt.plot(rw[fit_index], fcorrs[fit_index], 'b.', alpha = 0.5)
        plt.plot(rw, conts, 'g', label = 'Continuum')
        plt.plot(rw, cont_corr, 'g--', label = 'Best-fit Model')
        #plt.plot(w_to_fit, fs[fit_index], 'r.', alpha = 0.7)


        plt.xlim([1130., 1300])
        #plt.yscale('log')
        plt.ylim([0.*np.nanmedian(fs[good]), 3.*np.nanmedian(fs[good])])
        plt.xlabel("Wavelength (Angstroms)", fontsize = 18)
        plt.ylabel("Flux (" + r'erg s$^{-1}$' + ' ' + r'cm$^{-2}$' + ' ' + r'A$^{-1}$' + ')' , fontsize = 18)
        plt.legend(fontsize = 16, loc = 'upper right')
        plt.text(1150, 2.8*np.nanmedian(fs[good]), target, fontsize = 18)
        plt.text(1150, 2.6*np.nanmedian(fs[good]), 'Velocities: ' + '{:3.0f}'.format(vmw) + ', '  + '{:3.0f}'.format(vgal) + ' ' + r'km s$^{-1}$', fontsize = 18)
        plt.text(1150, 2.4*np.nanmedian(fs[good]), 'log N(HI): ' + '{:4.2f}'.format(best_hi_mw) + ', '  + '{:4.2f}'.format(best_hi_gal) + ' ' + r'cm$^{-2}$', fontsize = 18)

        fig.tight_layout()
        fig = plt.gcf()
        fig.savefig(outdir + outname + '_fitted_spectra.pdf', format = 'pdf', dpi = 1000)
        #plt.show()
        plt.clf()
        plt.close()



        return(nhi_mw_21cm, l_nhi_mw, best_hi_mw, best_fit_sigmalow_mw, best_fit_sigmahigh_mw, p50_mw, percentiles_sigmalow_mw,percentiles_sigmahigh_mw, l_nhi_gal, best_hi_gal, best_fit_sigmalow_gal, best_fit_sigmahigh_gal, p50_gal, percentiles_sigmalow_gal,percentiles_sigmahigh_gal, fs, chi2, rw, cont_corr, tau)



def fit_nhi_metali(rw,fs,es,conts,cont_std_index,l0_mw, l0_gal, fit_window_file,grid_mw, grid_gal, smooth = 5, outname = 'nhi_fitting', outdir = './', tau_max = 3., tau_min = 0.3, show=False, galaxy  = 'LMC'):

    """
    Fits a lorentzian profile to the Ly-a 1216 line to determine HI column densities in the galaxy observed and MW
    inputs are:
    ** rw: wavelength array
    ** fs: flux array
    ** es: error array
    ** cont_std_index: index relative to rw, fs, es of where to estimate the continuum
    ** l0_mw: wavlength of the center Ly-a MW absorption
    ** l0_gas: wavelength of the center Ly-a galaxy absorption
    ** fit_window_file: path to text file with the wavelength intervals to be used for fitting the profile
    ** grid_mw: 1D grid of N(HI) values for the MW component
    ** grid_gal: 1D grid of N(HI) values for the gal component

    OUTPUTS:
    best_hi_mw, mx_sigma_mw, p50_mw, c68_unc_mw, best_hi_gal, mx_sigma_gal, p50_gal, c68_unc_gal, fs, chi2
    ** best_hi_mw, gal: best fit N(HI) for MW, gal
    ** mx_sigma_mw, gal: error on N(HI) for MW, gal
    ** p50_mw, gal: N(HI) 50% value for MW, gal
    ** c68_unc_mw, gal: N(HI) value at 68% for MW, gal
    ** fs: input spectrum
    ** chi2: chi2!
    """

    l0 = np.double(1215.67)
    vmw = 3.e5*(l0_mw-l0)/l0
    vgal = 3.e5*(l0_gal-l0)/l0

    target = outname.split('_')[0]

    print("TAU_MAX ", tau_max)
    print("TAU_MIN ", tau_min)

    sampling_mw = np.median(np.abs(grid_mw-np.roll(grid_mw,1)))
    sampling_gal = np.median(np.abs(grid_gal-np.roll(grid_gal,1)))

    nmw = len(grid_mw)
    ngal = len(grid_gal)

    ln_prob = np.zeros([nmw, ngal], dtype = 'float32')
    chi2 = np.zeros([nmw, ngal], dtype = 'float32')

    if os.path.isfile(fit_window_file)==False:
        print("CREATE WINDOWS FOR CHI2 ESTIMATION")
        plt.plot(rw, fs, 'k')
        #plt.plot(rw, es*np.exp(3.), 'b')
        plt.xlim([1130, 1320])
        good = np.where((np.isnan(fs)==False) & (np.isinf(fs)==False))
        plt.ylim([0, 2.5*np.nanmedian(fs[good])])
        plt.show()

    t = ascii.read(fit_window_file)
    wmin = t['col1'].data
    wmax = t['col2'].data

    #implemented the N(HI) depdendent shift in teh scattering cross section from Lee+2003
    sigma_mw = 4.26e-20/(6.04e-10 + (np.double(rw)-l0_mw)**2)*(1.-1.792*(np.double(rw)-l0_mw)/l0_mw)
    sigma_gal = 4.26e-20/(6.04e-10 + (np.double(rw)-l0_gal)**2)*(1.-1.792*(np.double(rw)-l0_gal)/l0_gal)

    #fit_index= get_cont_index(wmin, wmax,rw)

    for i in range(nmw):
        for j in range(ngal):

            hi_mw = grid_mw[i]
            hi_gal = grid_gal[j]

            tau_mw = 10.**(hi_mw)*sigma_mw
            tau_gal= 10.**(hi_gal)*sigma_gal
            correction = np.exp(tau_mw)*np.exp(tau_gal)
            tau = np.log(correction)

            corr_f = fs*correction
            corr_e = es*correction

            #Fixed the windows, does nto depend on tau anymore
            fit_index = get_fit_index(rw, wmin, wmax, tau, tau_max = tau_max, tau_min = tau_min)

            if len(fit_index)>0:

                #print("OK ", grid_mw[i], grid_gal[j])

                spec_to_fit = fs[fit_index]
                err_to_fit = es[fit_index]
                tau_to_fit = tau[fit_index]
                w_to_fit = rw[fit_index]

                cont_to_fit = conts[fit_index]

                #if grid_gal[j]>=20.:
                #    plt.plot(w_to_fit, spec_to_fit, 'k.')
                #    plt.plot(w_to_fit, cont_to_fit, 'r.')
                #    this_model = cont_to_fit*np.exp(-tau_to_fit)
                #    plt.plot(w_to_fit, this_model, 'g.')
                #    plt.show()


                #for COS, teh gap can be nan, so filter it out

                valid = np.where((np.isnan(spec_to_fit) ==False) & (np.isinf(spec_to_fit)==False) & (err_to_fit > 0.))
                valid = valid[0]
                n = len(valid)

                chi2[i,j] = np.double(np.nansum((spec_to_fit[valid] - cont_to_fit[valid]*np.exp(-tau_to_fit[valid]))**2/err_to_fit[valid]**2))
                #print("DEBUG ", grid_mw[i], grid_gal[j], chi2[i,j])

            else:
                #print("PB ", grid_mw[i], grid_gal[j])
                chi2[i,j] = 100000000.

            ln_prob[i,j] = - chi2[i,j]/2.
    #the max is too high and turns into nan numerically, so subtract teh max, then normalize
    ln_prob = ln_prob-np.max(ln_prob)

    ln_prob = np.double(ln_prob) + np.log(1.) - np.log(np.sum(np.exp(np.double(ln_prob))))#- np.log(np.median(grid_mw-np.roll(grid_mw,1)))
    prob= np.exp(np.double(ln_prob))

    min_index = np.argmax(ln_prob)
    ij = np.unravel_index(min_index, ln_prob.shape)

    best_hi_mw = grid_mw[ij[0]]
    best_hi_gal = grid_gal[ij[1]]

    tau_mw = 10.**(best_hi_mw)*sigma_mw
    tau_gal= 10.**(best_hi_gal)*sigma_gal
    correction = np.exp(tau_mw)*np.exp(tau_gal)
    tau= np.log(correction)
    #fcorr = f*correction
    fcorrs = fs*correction
    cont_corr = conts*np.exp(-tau)
    bad = np.where(tau > 10)
    #fcorr[bad] = np.nan
    fcorrs[bad] = np.nan

    fit_index = get_fit_index(rw, wmin, wmax, tau, tau_max = tau_max,tau_min= tau_min)
    #fit_index = get_cont_index(wmin, wmax, w)
    spec_to_fit = fs[fit_index]
    w_to_fit = rw[fit_index]
    cont_to_fit = conts[fit_index]

    hdu = fits.PrimaryHDU(ln_prob)
    hdu2 = fits.ImageHDU(chi2 - np.nanmin(chi2))
    hdulist = fits.HDUList([hdu, hdu2])
    hdulist.writeto(outname + "_ln_prob.fits", overwrite=True)
    #bad = np.where((np.isnan(prob) ==True) | (np.isinf(prob)==True))
    #prob[bad] = 0.
    mx_pdf_mw = np.nanmax(prob, axis = 1)
    mx_pdf_gal = np.nanmax(prob, axis = 0)
    mx_pdf_mw = mx_pdf_mw/np.nansum(mx_pdf_mw)
    mx_pdf_gal = mx_pdf_gal/np.nansum(mx_pdf_gal)

    int_pdf_mw = np.nansum(prob, axis = 1)
    int_pdf_gal = np.nansum(prob, axis = 0)
    int_pdf_mw = int_pdf_mw/np.nansum(int_pdf_mw)
    int_pdf_gal = int_pdf_gal/np.nansum(int_pdf_gal)


    conf_level = 0.32 #corresponds to delta S = 2.3

    mx_sigma_mw = fwhm(grid_mw, mx_pdf_mw, level = conf_level, debug=False)
    mx_sigma_mw = mx_sigma_mw/2.
    #sigma_mw = fwhm_mw/np.sqrt(8.*np.log(2.))
    #sigma_mw = np.sqrt(np.sum(grid_mw**2*pdf_mw)/np.sum(pdf_mw) - exp_mw**2)

    mx_sigma_gal = fwhm(grid_gal, mx_pdf_gal, level = conf_level, debug=False)
    mx_sigma_gal = mx_sigma_gal/2.
    #sigma_gal = fwhm_gal/np.sqrt(8.*np.log(2.))
    #sigma_gal = np.sqrt(np.sum(grid_gal**2*pdf_gal)/np.sum(pdf_gal) - exp_gal**2)

    p50_mw, c68_unc_mw = get_p50(grid_mw, int_pdf_mw, level = 0.68)
    p50_gal, c68_unc_gal = get_p50(grid_gal, int_pdf_gal, level = 0.68)

    pdf_table_mw = Table()
    pdf_table_mw['N(HI)'] = grid_mw
    pdf_table_mw['MX_PDF'] = mx_pdf_mw
    pdf_table_mw['INT_PDF'] = int_pdf_mw
    pdf_mw_file = outdir + outname + '_mw_pdf.dat'
    ascii.write(pdf_table_mw, pdf_mw_file, format = 'csv', overwrite=True)

    pdf_table_gal = Table()
    pdf_table_gal['N(HI)'] = grid_gal
    pdf_table_gal['MX_PDF'] = mx_pdf_gal
    pdf_table_gal['INT_PDF'] = int_pdf_gal

    pdf_gal_file = outdir + outname + '_gal_pdf.dat'
    ascii.write(pdf_table_gal, pdf_gal_file, format = 'csv', overwrite=True)



    print("RESULT MW ", best_hi_mw, mx_sigma_mw, p50_mw, c68_unc_mw)
    print("RESULT " + galaxy, best_hi_gal, mx_sigma_gal, p50_gal, c68_unc_gal)

    plt.clf()
    plt.close()
    fig = plt.figure(figsize = (10,9))
    plt.plot(grid_mw, int_pdf_mw, 'k', label  = "MW")
    plt.plot(grid_gal, int_pdf_gal, 'r' , label = galaxy)
    plt.xlabel("log N(HI) (cm" + r'$^{-2}$)', fontsize = 18)
    plt.ylabel("PDF", fontsize = 18)
    plt.legend(fontsize = 18)
    plt.text(min(grid_mw) + 0.1, max(int_pdf_mw)*0.5, target, fontsize = 18)
    fig.tight_layout()
    fig = plt.gcf()
    fig.savefig(outdir + outname + "_pdfs.pdf", format = "pdf", dpi = 1000)
    #plt.show()

    plt.clf()
    plt.close()

    #fig,ax = plt.subplots(figsize = (11,9))
    #im=ax.imshow(prob, vmin = np.max(prob)/10., vmax = np.max(prob), origin = 'lower', extent = [np.min(grid_gal), np.max(grid_gal), np.min(grid_mw), np.max(grid_mw)], cmap = 'gist_stern_r')
    #ax.contour(grid_gal, grid_mw, prob, levels = [0.6*np.max(prob)], colors  = ['magenta'])
    #ax.set_xlabel("LMC log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
    #ax.set_ylabel("MW log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
    #ax.text(min(grid_gal) + 0.1, min(grid_mw)+0.1, target, fontsize = 18)
    #cbaxes = inset_axes(ax, width="80%", height="5%", loc=9)
    #cbar = fig.colorbar(im,cax=cbaxes, orientation='horizontal')
    ##cbar =fig.colorbar(im,orientation='vertical', fraction  = 0.04, ax = ax)
    #cbar.set_label("p = exp(-" + r'$\chi^2/2)$', fontsize = 18)
    #fig.subplots_adjust(bottom = 0.1, top = 0.95, left = 0.1, right = 0.85)
    #plt.savefig(outdir + outname + '_prob.pdf', format = 'pdf', dpi = 1000)
    #plt.clf()
    #plt.close()

    fig = plt.figure(figsize = (11,11))
    grid = plt.GridSpec(5, 5, hspace=0., wspace=0.)
    main_ax = fig.add_subplot(grid[:-1,1:])
    y_hist = fig.add_subplot(grid[:-1,0], xticklabels = [], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1,1:], yticklabels = [], sharex=main_ax)

    # scatter points on the main axes
    im = main_ax.imshow(prob, vmin = np.max(prob)/10., vmax = np.max(prob), origin = 'lower', extent = [np.min(grid_gal), np.max(grid_gal), np.min(grid_mw), np.max(grid_mw)], cmap = 'gist_stern_r')
    main_ax.contour(grid_gal, grid_mw, prob, levels = [conf_level*np.max(prob)], colors  = ['orange'])

    main_ax.text(min(grid_gal) + 0.1, min(grid_mw)+0.1, target, fontsize = 18)
    main_ax.set_xticklabels([])
    main_ax.set_yticklabels([])
    cbaxes = inset_axes(main_ax, width="80%", height="5%", loc=9)
    cbar = fig.colorbar(im,cax=cbaxes, orientation='horizontal')
    #cbar =fig.colorbar(im,orientation='horizontal', fraction  = 0.04, gap = -0.04, ax = main_ax)
    cbar.set_label("p ~ exp(-" + r'$\chi^2/2)$', fontsize = 18)

    #a zoom panel
    zax = plt.axes([0.65, 0.65, 0.2, 0.2])
    delta_mw = min([mx_sigma_mw/sampling_mw*5., 0.5/sampling_mw])
    delta_gal = min([mx_sigma_gal/sampling_gal*5.,0.5/sampling_gal])
    xmin_mw = int(max([ij[0]-delta_mw,0]))
    xmax_mw = int(min([ij[0] + delta_mw, nmw]))
    xmin_gal = int(max([ij[1]-delta_gal,0]))
    xmax_gal = int(min([ij[1] + delta_gal, ngal]))

    zax.imshow(prob[xmin_mw:xmax_mw, xmin_gal:xmax_gal], vmin = 0., vmax = np.max(prob), origin = 'lower', extent = [np.min(grid_gal[xmin_gal:xmax_gal]), np.max(grid_gal[xmin_gal:xmax_gal]), np.min(grid_mw[xmin_mw:xmax_mw]), np.max(grid_mw[xmin_mw:xmax_mw])], cmap = 'gist_stern_r')
    zax.contour(grid_gal[xmin_gal:xmax_gal], grid_mw[xmin_mw:xmax_mw], prob[xmin_mw:xmax_mw, xmin_gal:xmax_gal], levels = [conf_level*np.max(prob)], colors  = ['orange'])

    # histogram on the attached axes
    y_hist.plot(mx_pdf_mw,grid_mw, 'r')
    #y_hist.xaxis.tick_top()
    #y_hist.set_ylim([np.min(grid_mw), np.max(grid_mw)])
    y_hist.set_ylabel("MW log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
    y_hist.set_yticks(np.arange(19,22.5, 0.5))
    y_hist.set_yticklabels(np.arange(19,22.5, 0.5))
    y_hist.set_xlabel("PDF", fontsize = 18)
    y_hist.invert_xaxis()

    x_hist.plot(grid_gal,mx_pdf_gal, 'r')
    x_hist.set_xlabel(galaxy + " log N(HI) " + r'(cm$^{-2}$)', fontsize = 18)
    x_hist.set_xticks(np.arange(19,22.5, 0.5))
    x_hist.set_xticklabels(np.arange(19,22.5, 0.5))
    x_hist.set_ylabel("PDF", fontsize = 18)
    x_hist.invert_yaxis()

    fig.subplots_adjust(bottom = 0.1, top = 0.95, left = 0.1, right = 0.95)
    plt.savefig(outdir + outname + '_prob.pdf', format = 'pdf', dpi = 1000)
    plt.clf()
    plt.close()

    good = np.where((np.isnan(fs) ==False) & (np.isinf(fs)==False))
    good = good[0]

    fig = plt.figure(figsize = (15, 8))
    plt.plot(rw, fs, 'k', label = 'Spectrum')
    plt.plot(rw, es, '-', color = 'gray', label = 'Error')
    plt.plot(rw[cont_std_index], fs[cont_std_index], '.', color = 'magenta', label = 'Continuum est.')
    plt.plot(w_to_fit, fs[fit_index], 'b.',  label = r'$\chi^2$ minimzation range')
    #plt.plot(w_to_fit, spec_to_fit, 'b.', label = r'$\chi^2$ minimzation range')
    plt.plot(rw, fcorrs, 'r', alpha = 0.3, label = 'Reconstructed spectrum (x ' + r'e$^{\tau}$' + ')')
    plt.plot(rw[fit_index], fcorrs[fit_index], 'b.', alpha = 0.5)
    plt.plot(rw, conts, 'g', label = 'Continuum')
    plt.plot(rw, cont_corr, 'g--', label = 'Best-fit Model')
    #plt.plot(w_to_fit, fs[fit_index], 'r.', alpha = 0.7)


    plt.xlim([1130., 1300])
    #plt.yscale('log')
    plt.ylim([0.*np.nanmedian(fs[good]), 3.*np.nanmedian(fs[good])])
    plt.xlabel("Wavelength (Angstroms)", fontsize = 18)
    plt.ylabel("Flux (" + r'erg s$^{-1}$' + ' ' + r'cm$^{-2}$' + ' ' + r'A$^{-1}$' + ')' , fontsize = 18)
    plt.legend(fontsize = 16, loc = 'upper right')
    plt.text(1150, 2.8*np.nanmedian(fs[good]), target, fontsize = 18)
    plt.text(1150, 2.6*np.nanmedian(fs[good]), 'Velocities: ' + '{:3.0f}'.format(vmw) + ', '  + '{:3.0f}'.format(vgal) + ' ' + r'km s$^{-1}$', fontsize = 18)
    plt.text(1150, 2.4*np.nanmedian(fs[good]), 'log N(HI): ' + '{:4.2f}'.format(best_hi_mw) + ', '  + '{:4.2f}'.format(best_hi_gal) + ' ' + r'cm$^{-2}$', fontsize = 18)

    fig.tight_layout()
    fig = plt.gcf()
    fig.savefig(outdir + outname + '_fitted_spectra.pdf', format = 'pdf', dpi = 1000)
    #plt.show()
    plt.clf()
    plt.close()

    return(best_hi_mw, mx_sigma_mw, p50_mw, c68_unc_mw, best_hi_gal, mx_sigma_gal, p50_gal, c68_unc_gal, fs, chi2)



def aod_proc_log_old(spec_file, line,flambda, type_ascii = False, wavename= 'WAVELENGTH', errname = 'ERROR', target = '', smooth = 5, outdir = './', element = 'O I', vmin_line= 200., vmax_line = 330, show=True, spline=False):

        #Read the spectrum

        if type_ascii==False:
            specf = spectrum_fits_to_ascii(spec_file, wavename = wavename, errname = errname)
        else:
            specf = spec_file

        spec = ascii.read(specf)

        w = spec['WAVELENGTH']
        f = spec['FLUX']
        e  = spec['ERROR']

        #First identify velocity range from S II 1250
        #line_sii =1250.578

        #line_sii_key = '{:7.3f}'.format(line_sii)

        #vhelio_sii,cont_sii, rms_sii, fn_sii, rmsn_sii = cont_fit(w, f, line_sii , '/astro/dust_kg/jduval/METAL/NHI_FITTING/' + target + '_sii_windows.dat', smooth =13 , plt_vmin = -300., plt_vmax= 700., degree=3, outname = target +  '_sii_cont_fit',outdir   = outdir)

        #fns_sii = convolve(fn_sii, Box1DKernel(13))
        #en_sii= e/cont_sii
        #ens_sii = en_sii/np.sqrt(13)

        #signal_sii = np.where((1.-fns_sii > 3.*ens_sii))
        #signal_sii  = signal_sii[0]

        #plt.clf()
        #plt.close()
        #plt.plot(vhelio_sii, fn_sii, 'k', vhelio_sii, fns_sii, 'r')
        #plt.plot(vhelio_sii[signal_sii], fn_sii[signal_sii], 'b.')
        #plt.xlim([0,500])
        #plt.ylim([0,2])
        #plt.show()




        #fit the continuum

        line_key = '{:7.3f}'.format(line)

        tw = Table()
        tw['col1'] = np.array([-300., vmax_line])
        tw['col2'] = np.array([vmin_line, 700])
        ascii.write(tw, target + '_auto_cont_fit_' + line_key + '_windows.dat', overwrite=True)

        vhelio,cont, rms, fn, rmsn = cont_fit(w, f, line , target + '_auto_cont_fit_' + line_key + '_windows.dat', smooth = smooth, plt_vmin = -300., plt_vmax= 700., degree=9, outname = target +  '_' + line_key + '_cont_fit',outdir   = outdir, show = show, spline=spline)

        en= e/cont

        if smooth > 1:
            fn = convolve(fn, Box1DKernel(smooth))
            en = en/np.sqrt(smooth)


        aod = np.log(1./fn)

        #signal = np.where((vhelio>vmin_line) &  (vhelio < vmax_line) & (aod > 0.) & ((1.-fns)>2.*ens))
        #signal = signal[0]

        vrange = np.where((vhelio>vmin_line) &  (vhelio < vmax_line))
        vrange = vrange[0]

        this_vhelio = vhelio[vrange]
        this_spec = fn[vrange]
        this_err = en[vrange]


        pixel_scale = np.median(np.abs(this_vhelio - np.roll(this_vhelio,1)))
        structure_size = 20./pixel_scale

        emission_threshold_spectrum = (1.-this_spec)/this_err
        threshold =1.
        this_filtered_fn, this_mask = threshold_morph_spectrum(this_spec, emission_threshold_spectrum,  structure_size, value = 1. )

        filtered_aod= np.zeros_like(fn)
        filtered_aod[vrange] = np.log(1./this_filtered_fn)
        filtered_fn = np.zeros_like(fn) + 1.
        filtered_fn[vrange] = this_filtered_fn

        filtered_lognav = np.log10(filtered_aod) - flambda + 14.576
        lognav = np.log10(aod) - flambda + 14.576

        fgood = np.where(this_mask==True)
        fgood = fgood[0]
        len_good = len(fgood)

        signal = np.where(filtered_fn < 1.)
        signal = signal[0]

        if len_good>0:
            valid = np.where((np.isnan(filtered_lognav)==False) & (np.isinf(filtered_lognav)==False))
            valid  =valid[0]
            na = np.trapz(10.**(filtered_lognav[valid]), vhelio[valid])
            delta = max(vhelio[vrange]) - min(vhelio[vrange])
            #nl= len(signal)
            noise = np.where((vhelio>vmin_line-2*delta) & (vhelio < vmin_line-delta) & (aod > 0.))
            noise =noise[0]
            #noise = noise[0:nl]
            noise_na = np.sum(10.**(lognav[noise]))
            logna = np.log10(na)
            noise_logna = noise_na/na/np.log(10.)
            mn = min(filtered_lognav[signal])
            mx = max(filtered_lognav[signal])

        else:
            #noise = np.where((vhelio>vmin_line) & (vhelio < vmax_line) & (aod > 0.))
            #noise =noise[0]
            #signal = noise
            #na = np.trapz(10.**(lognav[noise]), vhelio[noise])
            #logna = np.log10(na)
            #noise_logna = 0.
            #mn = min(lognav[noise])

            na= 1.e10
            logna = np.log10(na)
            noise_na = 0.
            noise_logna = 0.
            mn= min(lognav[np.where((np.isnan(lognav)==False) & (np.isinf(lognav)==False))])
            mx = max(lognav[np.where((np.isnan(lognav)==False) & (np.isinf(lognav)==False))])

        print("COLUMN ", logna, noise_logna)

        plt.clf()
        plt.close()
        fig, ax = plt.subplots(2,sharex=True,figsize = (10,14))

        dumx= np.arange(-100,600,1)
        dumy = np.zeros(len(dumx)) +1.
        ax[0].plot(vhelio, fn, 'k')
        ax[0].plot(vhelio, filtered_fn, 'r')

        ax[0].plot(dumx, dumy, 'b--')
        ax[0].fill_between(vhelio[vrange], fn[vrange], facecolor = 'b', alpha = 0.3, linestyle = '--')
        ax[0].fill_between(vhelio[signal], fn[signal], facecolor = 'r', alpha= 0.3, linestyle = '--')
        ax[0].set_xticklabels([])
        ax[0].set_title(target, fontsize = 20)
        ax[0].set_ylim(bottom = 0, top = 1.5)
        ax[0].set_xlim(left =0, right = 500)


        ax[1].plot(vhelio, lognav, 'k')
        ax[1].plot(vhelio, filtered_lognav, 'r')
        ax[1].fill_between(vhelio[signal], mn, filtered_lognav[signal], facecolor  = 'r', linestyle = '--', alpha = 0.3)
        ax[1].set_xlabel("Heliocentric Velocity (km " + r's$^{-1}$' + ')', fontsize = 18)
        ax[1].set_ylabel("Column density (" + r'cm$^{-2}$' + ')', fontsize = 18)

        #plt.plot(vhelio, ens, 'r')
        ax[1].set_ylim(bottom = mn - 1.2 ,  top  =  mx + 1.)
        ax[1].text(100, mx + 0.6, element + ' ' + '{:6.2f}'.format(line), fontsize = 20)
        ax[1].text(100, mx + 0.2, "log N = " + '{:4.2f}'.format(np.log10(na)) + " " + r'$\pm$' + " " + '{:3.2f}'.format(noise_logna) + " " +  r'cm$^{-2}$', fontsize = 20)
        ax[1].set_xticks(np.arange(0, 500, 50))
        ax[1].set_xticklabels(np.arange(0, 500, 50))
        ax[1].set_xlim(left = 0, right = 500)

        fig.subplots_adjust(hspace = 0, bottom = 0.1, left = 0.1, top = 0.95, right = 0.95)
        fig = plt.gcf()
        fig.savefig(outdir + "plot_aod_" + target + '_' +  element.replace(' ', '') + "_" + line_key + '.pdf', format = 'pdf', dpi = 1000)
        if show:
            plt.show()
        plt.clf()
        plt.close()


        return(logna, noise_logna)
