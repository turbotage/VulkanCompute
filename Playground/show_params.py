import struct
from matplotlib import pyplot as plt
import numpy as np

import os

abs_path = os.path.dirname(__file__)
rel_path = 'data/ivim_params.vcdat'
full_path = os.path.join(abs_path, rel_path)

params_array = np.array(0)

with open(full_path, 'rb') as ifile:
	ba = bytearray(ifile.read())
	fa = np.empty(round(len(ba) / 4), dtype='float32')
	for i in range(round(len(ba) / 4)):
		fa[i] = struct.unpack('f', ba[4*i:(4*i+4)])[0]
	params_array = fa
	
print(params_array.shape)

params = np.empty([210-45,230-30,4], dtype='float32')

xn = params.shape[0]
yn = params.shape[1]

print(xn*yn*4)

for i in range(xn):
	for j in range(yn):
		k = 4 * i * yn + 4 * j
		params[i,j,0] = params_array[k+0]
		params[i,j,1] = params_array[k+1]
		params[i,j,2] = params_array[k+2]
		params[i,j,3] = params_array[k+3]


abs_path = os.path.dirname(__file__)

plt.figure(0)
plt.imshow(params[:,:,0].T, origin='lower', cmap='gray', interpolation='nearest')
rel_path = 'data/S0_image.png'
full_path = os.path.join(abs_path, rel_path)
fig = plt.gcf()
fig.savefig(full_path)


plt.figure(1)
plt.imshow(params[:,:,1].T, origin='lower', cmap='gray', interpolation='nearest')
rel_path = 'data/f_image.png'
full_path = os.path.join(abs_path, rel_path)
fig = plt.gcf()
fig.savefig(full_path)


plt.figure(2)
plt.imshow(params[:,:,2].T, origin='lower', cmap='gray', interpolation='nearest')
rel_path = 'data/d1_image.png'
full_path = os.path.join(abs_path, rel_path)
fig = plt.gcf()
fig.savefig(full_path)


plt.figure(3)
plt.imshow(params[:,:,3].T, origin='lower', cmap='gray', interpolation='nearest')
rel_path = 'data/d2_image.png'
full_path = os.path.join(abs_path, rel_path)
fig = plt.gcf()
fig.savefig(full_path)


