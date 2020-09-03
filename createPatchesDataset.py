import utils
import pandas as pd
import nibabel as nib
import numpy as np
import scipy
import random
import pickle
import argparse

def createDataset(df_patients, patchsize, num_patches, outputsize, outputfile):
    patches_per_patient = round(num_patches / len(df_patients))

    patch_num = 0
    patcheslist = []

    for i, row in df_patients.iterrows():
        volume = nib.load(row.volumepath)
        volume_data = volume.get_fdata()

        mask = nib.load(row.maskpath)
        mask_data = mask.get_fdata()

        volume_data = utils.norm_01(volume_data, mask_data)

        norm_patchsize_x = int(round(patchsize / volume.header['pixdim'][1]))
        norm_patchsize_y = int(round(patchsize / volume.header['pixdim'][2]))

        liver_loc = np.array(np.where(mask_data == 1))
        where_arr = np.array(np.concatenate(
            (np.where(liver_loc[0] < norm_patchsize_x), np.where(liver_loc[0] > mask.shape[0] - norm_patchsize_x),
             np.where(liver_loc[1] < norm_patchsize_y), np.where(liver_loc[1] > mask.shape[1] - norm_patchsize_y)),
            axis=1))
        liver_loc = np.delete(liver_loc, where_arr, axis=1)

        for j in range(patches_per_patient):
            try:
                p = random.randint(0, len(liver_loc[0]) - 1)
                x = liver_loc[0][p]
                y = liver_loc[1][p]
                z = liver_loc[2][p]

                patch = volume_data[x - norm_patchsize_x:x + norm_patchsize_x, y - norm_patchsize_y:y + norm_patchsize_y,
                        z]
                dsfactor = outputsize / (np.array([norm_patchsize_x * 2.0, norm_patchsize_y * 2.0]))
                patch = scipy.ndimage.zoom(patch, dsfactor)

                patcheslist.append(patch)
                patch_num += 1
            except IndexError:
                print('index error')

    with open(outputfile, 'wb') as f:
        pickle.dump(patcheslist, f)

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='Create a patches dataset.')
#    parser.add_argument('')
#    createDataset(some_arguments)