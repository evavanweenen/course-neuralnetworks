import numpy as np

root = '/disks/strw9/willebrandsvanweenen/assignment3/data/'

tab = np.loadtxt(root + 'merger_part1.txt', dtype='float', skiprows=1)

print(tab)

np.savetxt('ra_part1.txt', tab[:,1], '%5.8f')
np.savetxt('dec_part1.txt', tab[:,2], '%5.8f')
