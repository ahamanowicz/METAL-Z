# METAL-Z
METAL-Z paper I code and important files. Includes everything needed to reproduce the work in the paper (except for running the profile fits by Kirill)

**DATA**

Coadded spectra can be found in the Box-Folder: /METALZ/COADDS/
Kirill used different coadd methods for his profile fitting routine.
All files in this repo use "old" sightlines names, contrary to Lorenzo+2022 catalog names used in the paper (affects only Sextans-A). A list of the  "old" names and their Lorenzo equivalents can be found in METAL-Z_Lorenzo+2022.txt

**NOTEBOOKS**

Many functions used in the notebooks come either from Julia’s repository spectro (which I have downloaded and therefore might not have the newest version – check compatibility) or my own repo spec_utils_alex, both included here. Notebooks can be find in NOTEBOOKS directory.

**METAL-Z-line-measurements.ipynb**
1.	**HI fitting for all sightlines** requires: SII continuum windows to measure velocity components ( sightline_sii_windows.dat, in km/s, centered on MW), continuum fit windows file (sightline_lyman_alpha_windows.dat, in km/s  centered on MW), and windows for Ly-alpha winds fitting (sightline_lyman_alpha_fit_windows.dat, in restframe A (Angstorm)). Additional codes to help visualize and choose the windows: plot_line_vel.py, plot_line_vel_HI.py. Continuum widow files can be found in the HI-FIT folder

2.	**Continuum subtraction and creation of spectral**. For continuum correction to run, you need continuum-fit windows files, named sightline_line_lineNO_cont_win.txt. These files can be found in CONT-FIT folder. Ready Spectral Snippets are in DATA (*_cont.dat files). Fe II lines 1142, 1143, and 1144 are in the same file (called Fe_1142) because they are close together and have one continuum correction.
Lines are numbered as in the line_list.txt: Fe II: 0 (1142),1 (1143), 2 (1144) ,20 (1608) S II: 9 (1250), 10 (1253), 11 (1259).

The rest of the notebook is the exploration of the EW measurements and different issues we came across with the lines (LSF correction, blends etc). This analysis is not included in the paper.

**COS LSF and missing flux**
Due to the wide wings of the COS LSf, a percentage of the flux of the line might be missed if we choose to integrate an equivalent width over a certain velocity range. We can estimate what percentage of the flux is missing, that routine is incorporated into the calculations of equivalent width. 

a) **Equivalent width measurements**: Using the strong line - SII 1253 or Fe II 1144 we set the equivalent width integration limits. Alternatively, we use the HI profile towards a particular sightline (from LITTLE-THINGS VLA 21 cm maps). We use the same velocity range for all lines towards a single sightline. If the line is not detected, we measure EW within the same limits and  use 3 sigma (so 3 * error of Equivalent width measurement) as an upper limit. The automatized method for finding integration limits - calculates the equivalent width of the line in increasing velocity limits starting at the center. We expect the measurement to reach a plateu when we reach the continuum. For this data, it seems that the method is not convergent and strongly depends on the assumed parameters, like the max number of steps (how far away from the line we integrate), the step size. The method works better in less crowded parts of the spectrum (e.g. SII rather then Fe II). LSF percentage is calculated over the same velocity limits and saved to the measurements file.

b) **Special cases**
  - **Manual limits** especially for WLM (which was later removed from the sample) and some Sextans A sightlines, using the same limits and central velocity for all lines, didn't work, and I needed to add custom shifts.
  - **Masking**: In Sextans A Fe II 1142 overlaps with another line (MW ISM?), impacting the Equivalent width measurements. Manual masking helped exclude the overlapping line from the measurements.
  - **Stack plot** This notebook includes a version of a stack plot, showing all measured lines centered at the systemic velocity, with equivalent width measurements in the integration range marked.
  - **Fe blending in IC 1613**: Due to the velocity differences, in IC 1613 Fe 1143 blended with 1142 form the Milky Way. The process of "deblending" is described in the Appendix of the paper. We measured 1134 and 1144 from MW, and used Curve of Growth to estimate 1142 equivalent width. then we subtracted 1142 from IC 1613 1143, getting a deblended value of equivalent width. 

c) **Comparison with Ed's measurements**: Last bit ofg the notebook shows a comparison between results from my measurements with those done by Ed Jenkins.

