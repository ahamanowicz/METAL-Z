# METAL-Z
METAL-Z paper I code and important files. Includes everything needed to reproduce the work in the paper (except for running the profile fits by Kirill)

**DATA**

Coadded spectra can be found in the Box-Folder: /METALZ/COADDS/
Kirill used different coadd methods for his profile fitting routine.
All files in this repo use "old" sightlines names, contrary to Lorenzo+2022 catalog names used in the paper (affects only Sextans-A). A list of the  "old" names and their Lorenzo equivalents can be found in METAL-Z_Lorenzo+2022.txt

**NOTEBOOKS**

Many functions used in the notebooks come either from Julia’s repository spectro (which I have downloaded and therefore might not have the newest version – check compatibility) or my own repo spec_utils_alex, both included here. 

**METAL-Z-line-measurements.ipynb**
1.	HI fitting for all sightlines; requires: SII continuum windows to measure velocity components ( <sightline>_sii_windows.dat, in km/s, centered on MW), continuum fit windows file (<sightline>_lyman_alpha_windows.dat, in km/s  centered on MW), and windows for Ly-alpha winds fitting (<sightline>_lyman_alpha_fit_windows.dat, in restframe A (Angstorm)). Additional codes to help visualize and choose the windows: plot_line_vel.py, plot_line_vel_HI.py. Continuum widow files can be found in the HI-FIT folder

2.	Continuum subtraction and creation of spectral snippets. For continuum correction to run, you need continuum fit windows files, named <sightline>_line_<line-no>_cont_win.txt. These files can be found in CONT-FIT folder. Ready Spectral Snippets are in DATA (*_cont.dat files). Fe II lines 1142, 1143, and 1144 are in the same file (called Fe_1142) because they are close together and have one continuum correction.
Lines are numbered as in the line_list.txt: Fe II: 0 (1142),1 (1143), 2 (1144) ,20 (1608) S II: 9 (1250), 10 (1253), 11 (1259).
