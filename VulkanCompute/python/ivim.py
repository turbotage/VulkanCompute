import os

import numpy as np
import matplotlib.pyplot as plt

from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data

fraw, fbval, fbvec = get_fnames('ivim')

data = load_nifti_data(fraw)
bvals,bvecs = read_bvals_bvecs(fbval, fbvec)

z = 1
b=0
x1, x2 = 45, 210
y1, y2 = 30, 230
data_slice = data[x1:x2, y1:y2, z, :]

plt.imshow(data[x1:x2, y1:y2, z, b].T, origin='lower',
           cmap="gray", interpolation='nearest')
#plt.savefig("CSF_slice.png")
plt.close()

abs_path = os.path.dirname(__file__)
rel_path = 'export/ivim_data.vcdat'
full_path = os.path.join(abs_path, rel_path)

with open(full_path, 'wb') as of:
    for i in range(data_slice.shape[0]):
        for j in range(data_slice.shape[1]):
            of.write(data_slice[i,j,:].astype(np.float32).tobytes())

rel_path = 'export/ivim_bvals.vcdat'
full_path = os.path.join(abs_path, rel_path)

with open(full_path, 'wb') as of:
    of.write(bvals.astype(np.float32).tobytes())

