import numpy as np
import matplotlib.pyplot as plt
import spectro as spec
from astropy import convolution
from astropy.io import fits


def plot_line(n=0, wav='', flux='', vel=0, veldown=-200, velup=600):

    # n - the line index from the line_list.txt
    # wavelength and flux tables, vel - velocity of the system
    # velcut - the cut in velocity around the spectrum +/- velcut
    # lines = np.loadtxt("line_list.txt", dtype='str')
    # 1) choose the line

    linename = 'HI'
    line = float(n)

    # 2) convert to velocity, correct to LSR  and move to the galaxy rest frame

    v_helio = (wav - line) / line * 3e5
    v_lsr = spec.helio_to_lsr(v_helio, ra, dec)
    print(v_lsr, v_helio)
    # 3) catout the spectrum  +/- x around the line -check for the continuum

    i_min = min(np.where(v_lsr > veldown)[0])
    i_max = max(np.where(v_lsr < velup)[0])

    # limits as for contfit
    line_cut_vel = v_lsr[i_min:i_max]
    line_cut_flux = flux[i_min:i_max]

    # 4) plot the line

    plt.figure(1, figsize=(10, 8))
    plt.plot(line_cut_vel, line_cut_flux, c='k')
    plt.annotate(linename + " " + str(round(line, 2)), (0.7, 0.9),
        xycoords='axes fraction', fontsize=18)

    plt.xlim([veldown, velup])
    # plt.ylim([min(line_cut_flux), 1.2 * max(line_cut_flux)])

    plt.show()


v_IC1613, v_SexA, v_WLM, v_LeoP = -233, 324, -130, 264


# choose sigthile - m - number form the table


# coordinates
vel = v_SexA

ra, dec = -4.705822222222222, 152.7699583333333   
sightline = 'SEXTANS-A-SA3'

# spectrum file
box = '/Users/ahamanowicz/Library/CloudStorage/Box-Box/METALZ/COADDS/'
file = 'SEXTANS-A-SA3_COS_coadd_LPall.fits'

hdul = fits.open(box + file)
data = hdul[1].data

wav = data['WAVELENGTH']
fx = np.array(data['FLUX'])
err = np.array(data['ERROR'])
flux = convolution.convolve(fx, convolution.Gaussian1DKernel(3))

# 0 - wave, 1 - flux, 2 - error
# apply the lsr correction

N = 1215.67


plot_line(n=N, wav=wav, flux=flux, vel=vel, veldown=-15000, velup=15000)
