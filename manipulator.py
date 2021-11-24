import argparse
import copy
import os
import time
from tqdm import tqdm

import numpy as np
import PIL.Image
import torch

import clip
from wrapper import (FaceLandmarksDetector, Generator, 
                     VGGFeatExtractor, e4eEncoder, PivotTuning)
from projector import project 

class Manipulator():
    """Manipulator for style editing

    in paper, use 100 image pairs to estimate the mean for alpha(magnitude of the perturbation) [-5, 5]

    *** Args ***
    G : Genertor wrapper for synthesis styles
    device : torch.device
    lst_alpha : magnitude of the perturbation
    num_images : num images to process

    *** Attributes ***
    S :  List[dict(str, torch.Tensor)] # length 2,000
    styles : List[dict(str, torch.Tensor)] # length of num_images
                (num_images, style)
    lst_alpha : List[int]
    boundary : (num_images, len_alpha)
    edited_styles : List[styles]
    edited_images : List[(num_images, 3, 1024, 1024)]
    """
    def __init__(
        self, 
        G, 
        device, 
        lst_alpha=[0], 
        num_images=1, 
        start_ind=0, 
        face_preprocess=True,
        dataset_name=''
    ):
        """Initialize 
        - use pre-saved generated latent/style from random Z
        - to use projection, used method "set_real_img_projection"
        """
        assert start_ind + num_images < 2000
        self.W = torch.load(f'tensor/W{dataset_name}.pt')
        self.S = torch.load(f'tensor/S{dataset_name}.pt')
        self.S_mean = torch.load(f'tensor/S_mean{dataset_name}.pt')
        self.S_std = torch.load(f'tensor/S_std{dataset_name}.pt')

        self.S = {layer: self.S[layer].to(device) for layer in G.style_layers}
        self.styles = {layer: self.S[layer][start_ind:start_ind+num_images] for layer in G.style_layers}
        self.latent = self.W[start_ind:start_ind+num_images]
        self.latent = self.latent.to(device)
        del self.W
        del self.S

        # S_mean, S_std for extracting global style direction
        self.S_mean = {layer: self.S_mean[layer].to(device) for layer in G.style_layers}
        self.S_std = {layer: self.S_std[layer].to(device) for layer in G.style_layers}

        # setting
        self.face_preprocess = face_preprocess
        if face_preprocess:
            self.landmarks_detector = FaceLandmarksDetector()
        self.vgg16 = VGGFeatExtractor(device).module
        self.W_projector_steps = 200
        self.G = G
        self.device = device
        self.num_images = num_images
        self.lst_alpha = lst_alpha
        self.manipulate_layers = [layer for layer in G.style_layers if 'torgb' not in layer] 

    def set_alpha(self, lst_alpha):
        """Setter for alpha
        """
        self.lst_alpha = lst_alpha

    def set_real_img_projection(self, img, inv_mode='w', pti_mode=None):
        """Set real img instead of pre-saved styles
        Args : 
        - img : img directory or img file path to manipulate
            - face aligned if self.face_preprocess == True
            - set self.num_images
        - inv_mode : inversion mode, setting self.latent, self.styles
            - w : use W projector (projector.project)
            - w+ : use e4e encoder (wrapper.e4eEncoder)
        - pti_mode : pivot tuning inversion mode (wrapper.PivotTuning)
            - None
            - w : W latent pivot tuning
            - s : S style pivot tuning
        """
        assert inv_mode in ['w', 'w+']
        assert pti_mode in [None, 'w', 's']
        allowed_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']

        # img directory input
        if os.path.isdir(img):
            imgpaths = sorted(os.listdir(img))
            imgpaths = [os.path.join(img, imgpath) 
                        for imgpath in imgpaths 
                        if imgpath.split('.')[-1] in allowed_extensions]
        # img file path input
        else:
            imgpaths = [img]

        self.num_images = len(imgpaths)

        if inv_mode == 'w':
            targets = list()
            target_pils = list()
            for imgpath in imgpaths:
                if self.face_preprocess:
                    target_pil = self.landmarks_detector(imgpath)
                else:
                    target_pil = PIL.Image.open(imgpath).convert('RGB')
                target_pils.append(target_pil)
                w, h = target_pil.size
                s = min(w, h)
                target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                target_pil = target_pil.resize((self.G.G.img_resolution, self.G.G.img_resolution), 
                                                PIL.Image.LANCZOS)
                target_uint8 = np.array(target_pil, dtype=np.uint8)
                targets.append(torch.Tensor(target_uint8.transpose([2,0,1])).to(self.device))

            self.latent = list()
            for target in tqdm(targets, total=len(targets)):
                projected_w_steps = project(
                    self.G.G,
                    self.vgg16,
                    target=target,
                    num_steps=self.W_projector_steps, # TODO get projector steps from configs
                    device=self.device,
                    verbose=False,
                )
                self.latent.append(projected_w_steps[-1])
            self.latent = torch.stack(self.latent)
            self.styles = self.G.mapping_stylespace(self.latent)

        else: # inv_mode == 'w+'
            # use e4e encoder
            target_pils = list()
            for imgpath in imgpaths:
                if self.face_preprocess:
                    target_pil = self.landmarks_detector(imgpath)
                else:
                    target_pil = PIL.Image.open(imgpath).convert('RGB')
                target_pils.append(target_pil)

            self.encoder = e4eEncoder(self.device)
            self.latent = self.encoder(target_pils)
            self.styles = self.G.mapping_stylespace(self.latent)

        if pti_mode is not None: # w or s
            # pivot tuning inversion 
            pti = PivotTuning(self.device, self.G.G, mode=pti_mode)
            new_G = pti(self.latent, target_pils)
            self.G.G = new_G

    def manipulate(self, delta_s):
        """Edit style by given delta_style
        - use perturbation (delta s) * (alpha) as a boundary
        """
        styles = [copy.deepcopy(self.styles) for _ in range(len(self.lst_alpha))]

        for (alpha, style) in zip(self.lst_alpha, styles):
            for layer in self.G.style_layers:
                perturbation = delta_s[layer] * alpha
                style[layer] += perturbation
        return styles

    def manipulate_one_channel(self, layer, channel_ind:int):
        """Edit style from given layer, channel index
        - use mean value of pre-saved style
        - use perturbation (pre-saved style std) * (alpha) as a boundary
        """
        assert layer in self.G.style_layers
        assert 0 <= channel_ind < self.styles[layer].shape[1]
        boundary = self.S_std[layer][channel_ind].item()
        # apply self.S_mean value for given layer, channel_ind
        for ind in range(self.num_images):
            self.styles[layer][ind][channel_ind] = self.S_mean[layer][channel_ind]
        styles = [copy.deepcopy(self.styles) for _ in range(len(self.lst_alpha))]
        
        perturbation = (torch.Tensor(self.lst_alpha) * boundary).numpy().tolist()
       
        # apply one channel manipulation
        for img_ind in range(self.num_images):
            for edit_ind, delta in enumerate(perturbation):
                styles[edit_ind][layer][img_ind][channel_ind] += delta

        return styles

    def synthesis_from_styles(self, styles, start_ind, end_ind):
        """Synthesis edited styles from styles, lst_alpha
        """
        styles_ = list()
        for style in styles:
            style_ = dict()
            for layer in self.G.style_layers:
                style_[layer] = style[layer][start_ind:end_ind].to(self.device)
            styles_.append(style_)

        imgs = [self.G.synthesis_from_stylespace(self.latent[start_ind:end_ind], style_).cpu() 
                for style_ in styles_]
        return imgs


def extract_global_direction(G, device, lst_alpha, num_images, dataset_name=''):
    """Extract global style direction in 100 images
    """
    assert len(lst_alpha) == 2
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # lindex in original tf version
    manipulate_layers = [layer for layer in G.style_layers if 'torgb' not in layer] 

    # total channel: 6048 (1024 resolution)
    resolution = G.G.img_resolution
    latent = torch.randn([1,G.to_w_idx[f'G.synthesis.b{resolution}.torgb.affine']+1,512]).to(device) # 1024 -> 18, 512 -> 16, 256 -> 14
    style = G.mapping_stylespace(latent)
    cnt = 0
    for layer in manipulate_layers:
        cnt += style[layer].shape[1]
    del latent
    del style

    # 1024 -> 6048 channels, 256 -> 4928 channels
    print(f"total channels to manipulate: {cnt}")
    
    manipulator = Manipulator(G, device, lst_alpha, num_images, face_preprocess=False, dataset_name=dataset_name)

    all_feats = list()

    for layer in manipulate_layers:
        print(f'\nStyle manipulation in layer "{layer}"')
        channel_num = manipulator.styles[layer].shape[1]

        for channel_ind in tqdm(range(channel_num), total=channel_num):
            styles = manipulator.manipulate_one_channel(layer, channel_ind)
            # 2 * 100 images
            batchsize = 10
            nbatch = int(100 / batchsize)
            feats = list()
            for img_ind in range(0, nbatch): # batch size 10 * 2
                start = img_ind*nbatch
                end = img_ind*nbatch + batchsize
                synth_imgs = manipulator.synthesis_from_styles(styles, start, end)
                synth_imgs = [(synth_img.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8).numpy()
                            for synth_img in synth_imgs]
                imgs = list()
                for i in range(batchsize):
                    img0 = PIL.Image.fromarray(synth_imgs[0][i])
                    img1 = PIL.Image.fromarray(synth_imgs[1][i])
                    imgs.append(preprocess(img0).unsqueeze(0).to(device))
                    imgs.append(preprocess(img1).unsqueeze(0).to(device))
                with torch.no_grad():
                    feat = model.encode_image(torch.cat(imgs))
                feats.append(feat)
            all_feats.append(torch.cat(feats).view([-1, 2, 512]).cpu())

    all_feats = torch.stack(all_feats).numpy()

    fs = all_feats
    fs1=fs/np.linalg.norm(fs,axis=-1)[:,:,:,None]
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:] # 5*sigma - (-5)*sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None]
    fs3=fs3.mean(axis=1)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]

    np.save(f'tensor/fs3{dataset_name}.npy', fs3) # global style direction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('runtype', type=str, default='test')
    parser.add_argument('--ckpt', type=str, default='pretrained/ffhq.pkl')
    parser.add_argument('--face_preprocess', type=bool, default=True)
    parser.add_argument('--dataset_name', type=str, default='')
    args = parser.parse_args()
    
    runtype = args.runtype
    assert runtype in ['test', 'extract'] 

    device = torch.device('cuda:0')
    ckpt = args.ckpt
    G = Generator(ckpt, device)

    face_preprocess = args.face_preprocess
    dataset_name = args.dataset_name

    if runtype == 'test': # test manipulator
        num_images = 100
        lst_alpha = [-5, 0, 5]
        layer = G.style_layers[6]
        channel_ind = 501
        manipulator = Manipulator(G, device, lst_alpha, num_images, face_preprocess=face_preprocess, dataset_name=dataset_name)
        styles = manipulator.manipulate_one_channel(layer, channel_ind)
        start_ind, end_ind= 0, 10
        imgs = manipulator.synthesis_from_styles(styles, start_ind, end_ind)
        print(len(imgs), imgs[0].shape)

    elif runtype == 'extract': # extract global style direction from "tensor/S.pt"
        num_images = 100
        lst_alpha = [-5, 5]
        extract_global_direction(G, device, lst_alpha, num_images, dataset_name=dataset_name)
