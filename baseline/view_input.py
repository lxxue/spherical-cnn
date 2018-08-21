import os
import sys
from os.path import abspath, dirname
from mayavi import mlab
import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))

from spherical_cnn import models

basedir = "./m30_log"
val_file = "/home/lixin/data/s2cnn2/m30/train.tfrecord"
test_file = "/home/lixin/data/s2cnn2/m30/test.tfrecord"
layers = ["input", "label"]
ckptfile = "best.ckpt"


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """
    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def make_sgrid(b, alpha, beta, gamma):
    from lie_learn.spaces import S2

    theta, phi = S2.meshgrid(b=b, grid_type='Driscoll-Healy')
    sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
    sgrid = sgrid.reshape((-1, 3))

    R = rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum('ij,nj->ni', R, sgrid)

    return sgrid

bandwidth = 32
sgrid = make_sgrid(bandwidth, 0, 0, 0)
x, y, z = sgrid[..., 0], sgrid[..., 1], sgrid[..., 2]
x = x.reshape((2*bandwidth, 2*bandwidth))
y = y.reshape((2*bandwidth, 2*bandwidth))
z = z.reshape((2*bandwidth, 2*bandwidth))

m30_class_to_idx = {
    'airplane': 0,
    'bench': 1,
    "bookshelf": 2,
    "bottle": 3,
    "bowl": 4,
    "car": 5,
    "cone": 6,
    "cup": 7,
    "curtain": 8,
    "door": 9,
    "flower_pot": 10,
    "glass_box": 11,
    "guitar": 12,
    "keyboard": 13,
    "lamp": 14,
    "laptop": 15,
    "mantel": 16,
    "person": 17,
    "piano": 18,
    "plant": 19,
    "radio": 20,
    "range_hood": 21,
    "sink": 22,
    "stairs": 23,
    "stool": 24,
    "tent": 25,
    "tv_stand": 26,
    "vase": 27,
    "wardrobe": 28,
    "xbox": 29,
}


sgrid = make_sgrid(32, 0, 0, 0)
val = models.get_tfrecord_activations(basedir, val_file, layers, ckptfile=ckptfile)
#test = models.get_tfrecord_activations(basedir, test_file, layers, ckptfile=ckptfile)


print(val['input'].shape)
print(val['label'].shape)
val_data = val['input']
val_label = val['label']

#test_data = test['input']
#test_label = test['label']


#print(np.allclose(val_data, test_data))
#print(np.allclose(val_label, test_label))

for i in range(len(val_data)):
    # for j in range(6):
    k = np.random.randint(len(val_data))
    cls_name = list(m30_class_to_idx.keys())[list(m30_class_to_idx.values()).index(val_label[k])]
    mlab.figure(cls_name, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(2000, 1500))
    mlab.clf()
    mlab.mesh(x, y, z, scalars=val_data[k, ..., 0], colormap='coolwarm')
    mlab.show()

#print(input['input'][0])

