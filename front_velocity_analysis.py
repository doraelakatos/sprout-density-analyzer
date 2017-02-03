import fnmatch
import numpy as np
import os, sys, argparse


def halffind(array):
    maxcoord = np.argmax(array)
    maxval = array[maxcoord]

    if maxval == 0.0:
        halfval = 0.0
        res_out = 0.0
    else:
        arrayslice = array[maxcoord:]
        endarray = np.where(arrayslice < maxval/2)
    
        if len(endarray[0]) == 0:
            halfval = array.shape[0]
            res_out = arrayslice[-1]
        else:
            halfval = endarray[0][0] + maxcoord
            res_out = arrayslice[endarray[0][0]]
    
    return res_out, halfval


############## argparse ##################
p = argparse.ArgumentParser(description='front velocity analysis')
p.add_argument('-field', action='store', dest='field', type=str, required=True, help='give the number of the field')
args = p.parse_args()

Xfield = args.field

# generating file name
subfold = './imgproc_imgand/allres/'
templ_end = '__sprout_density_imgand.dat'

fname = ''

for fl in os.listdir(subfold):
    templ_fname = str(Xfield) + str(templ_end)
    
    if fnmatch.fnmatch(fl, '*'+str(templ_end)):
        if fnmatch.fnmatch(fl, '*'+str(templ_fname)):
            foldfi = fl[:-len(templ_fname)]
            fname = str(subfold) + str(foldfi) + str(templ_fname)
            print fname
            break

if len(fname) == 0:
    print "No such file or directory!"
    sys.exit(1)

########################### velocity calculation ##############################
# loading density dat file
# #1 radi, #2 area, #3 central density ratio, #4 t=1 ...

densfile = np.genfromtxt(fname, dtype=str)
densdat = densfile.astype(np.float)

# subtracting area of the aggregate
# velocity analysis starts from the 5th frame
radi = densdat[:, 0]
densprep = (densdat[:, 7:].T - densdat[:, 2].T).T

# finding values and locations belonging to maximal and half maximal densities
maxlocs_re = np.argmax(densprep[::-1, :], axis=0)
halfdens = np.apply_along_axis(halffind, 0, densprep)
maxlocs = (densprep.shape[0]-1)*np.ones((maxlocs_re.shape[0])) - maxlocs_re

dens_velo = np.zeros((maxlocs.size, 3))

for i in range(maxlocs.size):
    
    # time, radius - position of maxdensval, maximum density value
    dens_velo[i, :] = np.array([i+5, radi[maxlocs[i]], densprep[maxlocs[i], i]])
    
    if dens_velo[i, 2] == 0.0:

        dens_velo[i, 1] = 0.0

dens_velo = (np.around(dens_velo, decimals=5)).astype('|S8')


# saving maximal density values and positions as a function of time
svfname = str(subfold) + str(foldfi) + str(Xfield) + '__sprout_density_imgand_maxvelo.dat'
np.savetxt(svfname, dens_velo, fmt='%s', delimiter='\t', header='frame, radius, maxval')

# saving half maximal values with positions
halfname = str(subfold) + str(foldfi) + str(Xfield) + '__sprout_density_imgand_maxvelo_halfval.dat'
np.savetxt(halfname, halfdens.T, fmt='%s', delimiter='\t', header='half maximum value, position of half max')
