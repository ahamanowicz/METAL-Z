import numpy as np
import sys
import matplotlib.pyplot as plt
import spectro as spec
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy import convolution

def plot_line(n=0, wav='', flux='', vel=0, veldown=-200, velup=600, sightline='', Fe='True'):

    # n - the line index from the line_list.txt
    # wavelength and flux tables, vel - velocity of the system
    # velcut - the cut in velocity around the spectrum +/- velcut

    lines = np.loadtxt("line_list.txt", dtype='str')
    # 1) choose the line

    print(lines.T[1][n], lines.T[2][n], lines.T[3][n])
    linename = lines.T[1][n]
    line = float(lines.T[2][n])

    # 2) convert to velocity, correct to LSR  and move to the galaxy rest frame

    v_helio = (wav - line) / line * 3e5
    v_lsr = spec.helio_to_lsr(v_helio, ra, dec) # - vel

    # 3) catout the spectrum  +/- x around the line -check for the continuum

    i_min, i_max = min(np.where(v_lsr > veldown)[0]), max(np.where(v_lsr < velup)[0])
    # limits as for contfit
    line_cut_vel = v_lsr[i_min:i_max]
    line_cut_flux = flux[i_min:i_max]

    # 4) plot the line
    plt.figure(1, figsize=(12, 6))
    plt.plot(line_cut_vel, line_cut_flux, c='k')
    plt.annotate(linename+ " " + str(round(line, 2)), (0.7, 0.9), xycoords='axes fraction', fontsize=18)
    plt.annotate(sightline, (0.2, 0.9), xycoords='axes fraction', fontsize=18)

    if Fe == 'True':
        plt.axvline(0, ls='--', color='grey')
        plt.axvline((1143.226 - 1142.366) / 1142.366 * 3e5 + vel, ls='--', color='grey')
        plt.axvline((1144.938 - 1142.366) / 1144.938 * 3e5 + vel, ls='--', color='grey')

    plt.axvline(0, ls='--', color='r')
    plt.axvline(vel, ls='--', color='g')
    plt.xlim([veldown, velup])
    plt.ylim([min(line_cut_flux), 1.2 * max(line_cut_flux)])

    plt.show()


v_IC1613, v_SexA, v_WLM, v_LeoP = -233, 324, -130, 264

# choose sigthile - m - number form the table
M = int(sys.argv[1])

stars = np.loadtxt("metalZ-targets.txt", delimiter=',', dtype='str', skiprows=1)
ra_hex = stars.T[1][M]
dec_hex = stars.T[2][M]

c = SkyCoord(ra=ra_hex, dec=dec_hex, unit=(u.hourangle, u.deg))

# coordinates
vel = v_SexA
ra, dec = c.ra.degree, c.dec.degree

sightline = stars.T[0][M]
print(sightline)

# spectrum file
file = '/Users/ahamanowicz/Library/CloudStorage/Box-Box/METALZ/COADDS/' + sightline + '_COS_coadd.fits'
print(file)
hdul = fits.open(file)
data = hdul[1].data

wav = data['WAVELENGTH']
fx = np.array(data['FLUX'])
err = np.array(data['ERROR'])
flux = convolution.convolve(fx, convolution.Gaussian1DKernel(1))

# 0 - wave, 1 - flux, 2 - error
# apply the lsr correction

N = int(sys.argv[2])

plot_line(n=N,wav=wav, flux=flux, vel=vel, veldown=-200, velup=600, sightline=sightline, Fe='True')
