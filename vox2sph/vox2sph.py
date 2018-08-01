import numpy as np
import scipy.ndimage
from os.path import abspath, dirname
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
print(dirname(dirname(abspath(__file__))))
from spherical_cnn.util import sph_sample
from datasets.tf_modelnet import ModelNet

def vol2sph_all(dataset, n, mode):
    dsize = len(dataset)
    sph_all = np.zeros((dsize, n, n))
    phi, theta = sph_sample(n, mode=mode)
    for i, (vol, label) in enumerate(dataset):
        sph_all[i] = vol2sph(vol, n, phi, theat, mode=mode)

    return vol2sph
        


def vol2sph(vol, n, phi, theta, x0=None, mode='DH'):
    """ Occupancy grid to spherical function.

    Returns:
        f (2D array): spherical grid; first dim is theta (lon), second is phi (lat)
    """
    #phi, theta = sph_sample(n, mode=mode)
    # no need to do this over and over again
    if x0 is None:
        x0 = np.array(scipy.ndimage.measurements.center_of_mass(vol))

    if mode == 'HP':
        pg, tg = phi, theta
    else:
        pg, tg = np.meshgrid(phi, theta)
        pg, tg = pg.ravel(), tg.ravel()

    f = np.zeros((len(pg), 1))
    for i, (p, t) in enumerate(zip(pg, tg)):
        cells = ray_cell_intersection(x0, t, p, vol.shape[0])
        cells = [c for c in cells if vol[c]]
        if cells:
            f[i] = max([np.linalg.norm(x0-c) for c in cells])

    if mode != 'HP':
        f = f.reshape((n, n))

    return f


def ray_cell_intersection(x0, theta, phi, n):
    """ Return cells in n x n x n grid that intersects ray from x0 on theta, phi direction.

    phi=0 is the north pole.

    TODO: test!
    """
    assert 0 <= theta <= 2*np.pi
    assert 0 <= phi <= np.pi
    s, c = np.sin, np.cos
    # each intersection with planes {x, y, z}=i for i in [0,n) adds a point to the list
    # rays:
    # x0 + t sin(phi)cos(theta)
    # y0 + t sin(phi)sin(theta)
    # z0 + t cos(phi)
    cells = []
    for i in range(n):
        # plane x=i
        t = (i-x0[0]) / (s(phi)*c(theta))
        if 0 <= t < n:
            y = int(t*s(phi)*s(theta) + x0[1])
            z = int(t*c(phi) + x0[2])
            cells.append((i, y, z))

        # plane y=i
        t = (i-x0[1]) / (s(phi)*s(theta))
        if 0 <= t < n:
            x = int(t*s(phi)*c(theta) + x0[0])
            z = int(t*c(phi) + x0[2])
            cells.append((x, i, z))

        # plane z=i
        t = (i-x0[2]) / c(phi)
        if 0 <= t < n:
            x = int(t*s(phi)*c(theta) + x0[0])
            y = int(t*s(phi)*s(theta) + x0[1])
            cells.append((x, y, i))

    cells = [c for c in cells if ((0 <= c[0] < n) and
                                  (0 <= c[1] < n) and
                                  (0 <= c[2] < n))]

    cells = list(set(cells))

    return cells

if __name__ == "__main__":
    dset = ModelNet("/home/lixin/Documents/github/spherical-cnn/data/", "shapenet10_test.tar")
    #print(len(dset))
    #print(dset[0])
    # data, label
    #print(dset[0][0].shape)
    #print("finishing loading dataset")
    sph_all = vol2sph_all(dset, n=32, mode='DH') 
