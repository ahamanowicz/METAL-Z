# METAL-Z
METAL-Z paper I code and essential files. Includes everything needed to reproduce the work in the paper (except for running the profile fits by Kirill)

**Requirements:** numpy, matplotlib, astropy, scipy, math, uncertanities (any version of numpy should work)

**DATA**

METAL-Z results file (all results used in the paper) are in Box: /METALZ/DATA/METALZ-res_PF.fits. The curve of growth measurements, also in the Box, are in METALZ-results.fits
In the same folder are D/G measurements for DLAs, which are used for the D/G-metallicity plot.

Coadded spectra can be found in the Box-Folder: /METALZ/COADDS/
Spectral snippets are available in Box-Folder: /METALZ/METAL_Z_voigt 
Kirill used different coadd methods for his profile fitting routine.
All files in this repo use "old" sightlines names, contrary to Lorenzo+2022 catalog names used in the paper (affects only Sextans-A). A list of the "old" names and their Lorenzo equivalents can be found in METAL-Z_Lorenzo+2022.txt.

**NOTEBOOKS**

Many functions used in the notebooks come from Julia's repository spectro (which I have downloaded and might not have the newest version – check compatibility) or my repo spec_utils_alex, both included here. Notebooks can be found in the NOTEBOOKS directory.


**METAL-Z full analysis for the paper-profile fit.ipynb**

Main notebook containing analysis of the results of METAL-Z measurements. Depletion measurements in this notebook are based on the profile fitting,  as in the paper.
To run, this notebook requires custom repositories (all included in /REPOS/): depletion_utils_alex, CogFuncątions, spectro, spec_utils_alex, 
and data tables: line_list, METAL-Z_res_PF, reference_abundances, metal-fe-s, dep_MW, dep-nh-MW, and depletions results from METAL for LMC, SMC, and MW. 

1. Contains all information about galaxies (velocities, redshifts) and lines used in the project.
2. Scales the abundances of alpha and Fe peak elements from known stellar abundances in IC 1613 and Sextans A
3. Calculates depletions from provided column densities
4. Create all plots and tables (in Latex form) for the paper

Create figures for the paper (all in Box folder METALZ/FIGURES):

a) Relation between d(Fe) and d(S): provides the plot (plot METAL relations for comparison,  requires metal-fe-s.txt)

b) N(SII) - N(H)  

c) Fe and S depletions relation with N(H). Requires fit coefficients calculated in Bayesian errors of the log(NH)-d(X).ipynb (here hardcoded). Plots METAL data points (requires metal measurement files). No fit for depletions of S.

d) As explained in the notebook and paper, to get D/G, we need to make assumptions about the behavior of depletions of other elements. We assume that the depletion relations between elements found in MW also apply to METAL-Z galaxies. 
Implementation: scale depletions of Fe to other elements using MW relations between d(fe) and d(X) (translate through F*), reproduce the relation with gas density for all elements, sum contribution of different elements to calculate D/G relation with gas density, find D/G at specific column densities. It also provides a plot with derived relations between depletion and column density for Si and Mg in IC 1613 and Sextans A.
Requires coefficients for relations between Fe and other elements in MW (dep_MW.txt)

e) D/G - metallicity plot, or Feldman plot, requires literature datasets (included in here, plot adapted from Julia): D/G from METAL (metal-dg.txt), DLAs (decia2016_table6.dat) together with scaling (LMC_fit_DG_NH.dat), FIR (remy-ruyer2014_dust_gas_masses.dat, dustpedia_combined_sample.csv). (DLAs might also require *.fits data from deCia and Quiret, which are included in the Box) 

f) D/G but only for Fe - without assumptions of other elements.

g) Other tables and plots used in the paper or Appendix

h) Alternative versions of plots for the HST proposal

**METAL-Z-line-measurements.ipynb**
1.	**HI fitting for all sightlines** requires: SII continuum windows to measure velocity components ( sightline_sii_windows.dat, in km/s, centered on MW), continuum fit windows file (sightline_lyman_alpha_windows.dat, in km/s  centered on MW), and windows for Ly-alpha winds fitting (sightline_lyman_alpha_fit_windows.dat, in restframe A (Angstorm)). Additional codes to help visualize and choose the windows: plot_line_vel.py, plot_line_vel_HI.py. Continuum widow files can be found in the HI-FIT folder

2.	**Continuum subtraction and creation of spectral**. For continuum correction, you need continuum-fit windows files, named sightline_line_lineNO_cont_win.txt. These files can be found in the CONT-FIT folder. Ready Spectral Snippets are in DATA (*_cont.dat files). Fe II lines 1142, 1143, and 1144 are in the same file (called Fe_1142) because they are close together and have one continuum correction.
Lines are numbered as in the line_list.txt: Fe II: 0 (1142),1 (1143), 2 (1144),20 (1608) S II: 9 (1250), 10 (1253), 11 (1259).

The rest of the notebook explores the EW measurements and the issues we encountered with the lines (LSF correction, blends, etc). This analysis is not included in the paper.

**COS LSF and missing flux**
Due to the broad wings of the COS LSf, a percentage of the flux of the line might be missed if we choose to integrate an equivalent width over a specific velocity range. We can estimate what percentage of the flux is missing; that routine is incorporated into the calculations of equivalent width. 

a) **Equivalent width measurements**: We set the equivalent width integration limits using the strong line - SII 1253 or Fe II 1144. Alternatively, we use the HI profile towards a particular sightline (from LITTLE-THINGS VLA 21 cm maps). We use the same velocity range for all lines towards a single sightline. If the line is undetected, we measure EW within the same limits and use three sigma (3 * error of Equivalent width measurement) as an upper limit. The automatized method for finding integration limits - calculates the equivalent width of the line in increasing velocity limits starting at the center. We expect the measurement to reach a plateau when we reach the continuum. For this data, the method seems not convergent and strongly depends on the assumed parameters, like the max number of steps (how far away from the line we integrate) and the step size. The method works better in less crowded parts of the spectrum (e.g., SII rather than Fe II). LSF percentage is calculated over the same velocity limits and saved to the measurements file.

b) **Special cases**
  - **Manual limits** especially for WLM (later removed from the sample) and some Sextans A sightlines, using the same limits and central velocity for all lines, didn't work, and I needed to add custom shifts.
  - **Masking**: In Sextans A Fe II 1142 overlaps with another line (MW ISM?), impacting the Equivalent width measurements. Manual masking helped exclude the overlapping line from the measurements.
  - **Stack plot** This notebook includes a version of a stack plot, showing all measured lines centered at the systemic velocity, with equivalent width measurements in the integration range marked.
  - **Fe blending in IC 1613**: Due to the velocity differences, IC 1613 Fe 1143 blended with 1142 from the Milky Way. The process of "deblending" is described in the paper's Appendix. We measured 1134 and 1144 from MW and used Curve of Growth to estimate 1142 equivalent width. Then, we subtracted 1142 from IC 1613 1143, getting a deblended value of equivalent width. 

c) **Comparison with Ed's measurements**: The last bit of the notebook compares results from my measurements with those done by Ed Jenkins.

**COG-example.ipnyb**

Shows how the Curve of Growth code works in the Prochaska et al. 2006 example.
The CoG process requires a model ( a grid of parameters), which can be generated using cog_model.py (in CODES/). You have to create a grid corresponding to that model in the notebook. To make CoG calculations, you need equivalent width with errors, line wavelengths, and f-values. The notebook creates a 'banana plot' and 2D probability maps for log(N) and b. 
The notebook uses the custom repository CogFunctions (in REPOS/).

**COG measurements METAL-Z.ipynb**

Similar to the example, but CoG calculations for METAL-Z measurements. Requires lines_list.txt, model.fits, table of equivalent width measurements. Provides banana plot (contours), 2D probability map, and final N and b values with errors. The last cell in the notebook is a quick loop over all sightlines.
Details about the design of the Durve of Growth calculations are described in the METAL-Z paper Appendix.

**Bayesian errors of the log(NH)-d(X).ipynb**

Bayesian linear regression of the log(NH) - d (X) relation, with errors. Requires input file with depletion measurements. 
Uses a and b parameter grid to find the most probable function parameters. All necessary functions are included in the notebook.
There are two versions of prior and posterior functions - the original ones are locked at the LMC value. New ones (called 'free') let both parameters be fitted. Funciton calc_zh calculates the z parameter (see RD+2021,2022a for function details). 
Calculate the best-fitting parameters for each galaxy and each element. The result is a plot of probabilities in the phase space of a (slope) and b (intercept). Code provides the best value of a and b parameters together with errors. We have fitting only for METAL-Z for Fe as most S depletions were positive. Most up-to-date solutions are at the top of the notebook, and later, different variations are included (with CoG measurements, with fixed b, etc.).
