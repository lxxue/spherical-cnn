import numpy as np
import sys
import os
from os.path import abspath, dirname
import re

sys.path.append(dirname(abspath(__file__)))
import npytar

class ModelNet():
    def __init__(self, filedir, filename):
        reader = npytar.NpyTarReader(os.path.join(filedir, filename))
        temp_list = []
        for (arr, name) in reader:
            temp_list.append((arr, name))
        reader.close()
        # in this dataset we have 12 views for each shape
        if len(temp_list) % 12 != 0:
            # assert is a statement in python2/3, not a function
            assert "some shapes might not have 12 views"
        # b x v x c x d x d x d
        #self.data = np.zeros((len(temp_list)//12, 12, 1, 32, 32, 32), dtype=np.float32)
        # b x v x d x d x d
        self.data = np.zeros((len(temp_list)//12, 12, 32, 32, 32), dtype=np.float32)
        # all view share the same label
        self.label = np.zeros((len(temp_list)//12,), dtype=np.int)
        # sort the file by its name
        # format: classnum.classname_idx.viewidx
        # exception: 001.2h31k8fak.viewidx
        temp_list.sort(key=lambda x: (int(x[1].split(".")[0]), x[1].split(".")[-2].split("_")[-1], int(x[1].split(".")[-1])))
        for idx, (arr, name) in enumerate(temp_list):
            #self.data[idx//12, idx%12, 0] = arr
            # no channel dim now
            self.data[idx//12, idx%12] = arr
            if idx % 12 == 0:
                # assign label
                # name: class_idx.fname.view_idx
                self.label[idx//12] = int(name.split('.')[0])-1
            else:
                # check label consistency
                assert self.label[idx//12]==(int(name.split('.')[0])-1), "label is inconsistent among different views for file {}, original label{}".format(name, self.label[idx//12])
        #finish loading all data

    def __len__(self):
        return self.label.shape[0]*12

    def __getitem__(self, idx):
        return self.data[idx//12, idx%12], self.label[idx//12]


if __name__ == "__main__":
    dset = ModelNet("/home/lixin/Documents/github/spherical-cnn/data/", "shapenet10_test.tar")
    print(len(dset))
    print(dset[0])
    # data, label
    print(dset[0][0].shape)
