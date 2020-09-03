import torch
from torch.utils.data.dataset import Dataset
import pickle
import nibabel as nib
import numpy as np
import utils
import scipy

class LiverPatchDS(Dataset):
    def __init__(self, patchespickle, labels=None):
        self.labels = labels

        with open(patchespickle, 'rb') as picklefile:
            self.patcheslist = pickle.load(picklefile)


    def setlabels(self, labels):
        self.labels = labels

    def __getitem__(self, index):
        patch = self.patcheslist[index]

        if len(patch.shape) < 2:
            patch = patch[None, :, :]

        if self.labels is None:
            return (torch.tensor(patch, dtype=torch.float), 0)

        return (torch.tensor(patch, dtype=torch.float), torch.tensor(self.labels[index]).type(torch.LongTensor))

    def __len__(self):
        return len(self.patcheslist)


class LiverDataset(Dataset):
    def __init__(self, imgpath, maskpath, patchsize, outsize, sample_step=1, norm=False):
        self.maskpath = maskpath
        self.imgpath = imgpath
        self.patchsize = patchsize
        self.outsize = outsize
        self.sample_step = sample_step
        self.norm = norm

        self.img = nib.load(self.imgpath)
        self.img_data = self.img.get_data()

        mask = nib.load(self.maskpath)
        mask_data = mask.get_data()

        self.img_data = utils.norm_01(self.img_data, mask_data)

        self.norm_patchsize_x = int(round(self.patchsize / self.img.header['pixdim'][1]))
        self.norm_patchsize_y = int(round(self.patchsize / self.img.header['pixdim'][2]))

        self.locations = self.get_liver_locations()

    def get_liver_locations(self):
        mask = nib.load(self.maskpath)
        mask_data = mask.get_data()

        if np.sign(mask.affine[1][1]) != np.sign(self.img.affine[1][1]):
            mask_data = np.fliplr(mask_data).copy()

        if self.sample_step != 1:
            arr = np.zeros((mask.shape))
            x = np.arange(0, mask.shape[0], self.sample_step)
            y = np.arange(0, mask.shape[1], self.sample_step)
            z = np.arange(0, mask.shape[2], 1)

            xx, yy, zz = np.meshgrid(x, y, z, sparse=True)
            arr[xx, yy, zz] = 1
            arr[mask_data == 0] = 0
            mask_data[arr != 1] = 0

        liver_loc = np.array(np.where(mask_data == 1))

        where_arr = np.array(np.concatenate((np.where(liver_loc[0] < self.norm_patchsize_x),
                                             np.where(liver_loc[0] > mask.shape[0] - self.norm_patchsize_x),
                                             np.where(liver_loc[1] < self.norm_patchsize_y),
                                             np.where(liver_loc[1] > mask.shape[1] - self.norm_patchsize_y)), axis=1))

        liver_loc = np.delete(liver_loc, where_arr, axis=1)

        return liver_loc

    def __getitem__(self, index):
        x = self.locations[0][index]
        y = self.locations[1][index]
        z = self.locations[2][index]

        patch = self.img_data[x - self.norm_patchsize_x:x + self.norm_patchsize_x,
                y - self.norm_patchsize_y:y + self.norm_patchsize_y, z]

        dsfactor = self.outsize / (np.array([self.norm_patchsize_x, self.norm_patchsize_y]) * 2.0)

        patch = scipy.ndimage.zoom(patch, dsfactor)

        if self.norm:
            patch = self.norm_01(patch)

        patch_as_tensor = torch.Tensor(patch)
        patch_as_tensor = patch_as_tensor[None, :, :]

        return (patch_as_tensor.type(torch.FloatTensor), self.locations[:, index])

    def __len__(self):
        return self.locations.shape[1]