import numpy as np
from astropy.io import fits
### multiprpocessing
import multiprocessing as mp
#print("Number of processors: ", mp.cpu_count())
from multiprocessing import Pool
import CogFunctions as cog

import time

def cog_model(Narray=np.arange(10,20,0.1), barray=np.arange(5,100,.5) , LFarray=np.linspace(-8,-4,100)):
	# ##generate the model array
	Model_array=np.zeros((np.size(N),np.size(b), np.size(LF_m)))
	for i in range(np.size(N)):
		for j in range(np.size(b)):

			Model_array[i][j] = np.log10(cog.CoG_full(N[i],b[j],LF_m))
	return Model_array

# ## create a model
# #grid for the model
N = np.arange(10,20,0.05)#logN 10 - 20 
b = np.arange(5,100,0.5)  # 5 -100
LF_m = np.linspace(-8,-4,100)
t0 = time.time()


result_array=cog_model(N,b, LF_m)

# # save the model
hdu = fits.PrimaryHDU(result_array.T)
hdul=fits.HDUList([hdu])
hdul.writeto('model_new.fits', overwrite=True)

t1 = time.time()
total = t1-t0

print("Elapsed time:", total)