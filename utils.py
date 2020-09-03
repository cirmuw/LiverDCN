import numpy as np
import models
import datasets
import torch
import nibabel as nib
import pickle

def norm_01(img_data, mask_data=None):
    img_data = img_data.astype('float')

    if mask_data is not None:
        maxi = np.max(img_data[mask_data==1])
        mini = np.min(img_data[mask_data==1])
    else:
        maxi = np.max(img_data)
        mini = np.min(img_data)

    r = (img_data - mini).astype(float)
    r = r / maxi

    return r


def get_signature_for_volume(modelpath, centerpath, volumepath, maskpath, savepath=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.autoencoder(inchannels=1, num_bottleneck=50).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    centers = np.load(centerpath, allow_pickle=True)

    ds = datasets.LiverDataset(volumepath, maskpath, patchsize=7, sample_step=2, norm=True)
    dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=16, num_workers=2)

    constructed_vol = np.zeros(ds.img.shape)
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            patches, position = data
            patches = patches.to(device)

            latent = model.get_latent(patches)
            latents = latent.detach().cpu().numpy()

            for j, l in enumerate(latents):
                dists = [np.sqrt(np.sum((l-c)**2)) for c in centers]
                constructed_vol[position[j][0]-1:position[j][0]+1, position[j][1]-1:position[j][1]+1, position[j][2]] = np.argmin(dists)+1

    nib_vol = nib.Nifti1Image(constructed_vol, ds.img.affine)
    if savepath is not None:
        nib.save(nib_vol, savepath)

    mask_voxels = np.count_nonzero(constructed_vol)
    value, count = np.unique(constructed_vol, return_counts=True)
    signature = np.zeros(len(centers))

    for i, v in enumerate(value):
        if v!=0:
            signature[int(v-1)] = float(count[i]/mask_voxels)

    return signature