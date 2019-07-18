import numpy as np
import wget
from astropy.table import Table, Column
from astropy import units as u
from astropy.coordinates import SkyCoord, angles

root = '/disks/strw9/willebrandsvanweenen/assignment3/data/'

merger1 = np.loadtxt(root + 'merger_part1.txt', dtype='float', skiprows=1) #merger galaxies
merger2 = np.loadtxt(root + 'merger_part2.txt', dtype='float', skiprows=1) #merger galaxies

data = Table.read(root + 'GalaxyZoo1_DR_table7.csv', format='csv') #GalaxyZoo

#right ascension in hourangle to degree
h = []
m = []
s = []
for row in data['RA']:
    h.append(row[:2])
    m.append(row[3:5])
    s.append(row[6:10])
h = np.array(h, dtype=np.float)
m = np.array(m, dtype=np.float)
s = np.array(s, dtype=np.float)
hfrac = h + m/60. + s/3600.
ra = hfrac/24.*360.

#declination in hourangle to degree
pn = []
d = []
am = []
ass = []
for row in data['DEC']:
    if row[:1] == '+':
        pn.append(1)
    elif row[:1] == '-':
        pn.append(-1)    
    d.append(row[1:3])
    am.append(row[4:6])
    ass.append(row[7:10])

d = np.array(d, dtype=np.float)
am = np.array(am, dtype=np.float)
ass = np.array(ass, dtype=np.float)
dec = (d + am/60. + ass/3600.)*pn

#Create random array of nonmergers
length = (data['RA'].shape)[0]
array= np.arange(0, length) #0,..,800.000

ra = np.around(ra, 5)
merger1[:,1] = np.around(merger1[:,1],5)
merger2[:,1] = np.around(merger2[:,1],5)
dec = np.around(dec, 7)
merger1[:,2] = np.around(merger1[:,2],7)
merger2[:,2] = np.around(merger2[:,2],7)

choice = np.random.choice(array, 10000, replace=False) #10.000 unieke getallen uit array
choice_nomerg = []

bo = False
for nr, i in enumerate(choice):
    if nr%1000 == 0:
        print(nr, len(choice_nomerg))
    for j in range(len(merger1)):
        if (ra[i] == merger1[j,1] and dec[i] == merger1[j,2]) and (ra[i] == merger2[j,1] and dec[i] == merger2[j,2]):
            bo = True
    if bo == False:
        choice_nomerg.append(i)
    bo = False
print(len(choice), len(choice_nomerg))

#Download images of nonmergers
for i in choice_nomerg:
    wget.download('http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra={0}&dec={1}&scale=0.5&width=100&height=100'.format(ra[i], dec[i]), out=root+'/random_galaxies/im{0}ra{1}dec{2}.png'.format(i, ra[i], dec[i]))
    print(i)

#Download images of mergers
for i in range(len(merger1[:,1])):
    wget.download('http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra={0}&dec={1}&scale=0.5&width=100&height=100'.format(merger1[i,1], merger1[i,2]), out=root+'/cut_final2/im{0}ra{1}dec{2}.png'.format(i, merger1[i,1], merger1[i,2]))
    print(i)

for i in range(len(merger2[:,1])):
    wget.download('http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra={0}&dec={1}&scale=0.5&width=100&height=100'.format(merger2[i,1], merger2[i,2]), out=root+'/cut_final2/im{0}ra{1}dec{2}.png'.format(i, merger2[i,1], merger2[i,2]))
    print(i)
