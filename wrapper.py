import time
import shutil

import dlib
import numpy as np
import PIL.Image 
import torch
from torchvision.transforms import transforms

import dnnlib
import legacy
from configs import GENERATOR_CONFIGS
from dlib_utils.face_alignment import image_align
from dlib_utils.landmarks_detector import LandmarksDetector
from torch_utils.misc import copy_params_and_buffers

from pivot_tuning_inversion.utils.ImagesDataset import ImagesDataset, ImageLatentsDataset
from pivot_tuning_inversion.training.coaches.multi_id_coach import MultiIDCoach


class FaceLandmarksDetector:
    """Dlib landmarks detector wrapper
    """
    def __init__(
            self, 
            model_path='pretrained/shape_predictor_68_face_landmarks.dat', 
            tmp_dir='tmp'
        ):

        self.detector = LandmarksDetector(model_path)
        self.timestamp = int(time.time())
        self.tmp_src = f'{tmp_dir}/{self.timestamp}_src.png'
        self.tmp_align = f'{tmp_dir}/{self.timestamp}_align.png'

    def __call__(self, imgpath):
        shutil.copy(imgpath, self.tmp_src)
        try:
            face_landmarks = list(self.detector.get_landmarks(self.tmp_src))[0]
            assert isinstance(face_landmarks, list)
            assert len(face_landmarks) == 68
            image_align(self.tmp_src, self.tmp_align, face_landmarks)
        except:
            im = PIL.Image.open(self.tmp_src)
            im.save(self.tmp_align)
        return PIL.Image.open(self.tmp_align).convert('RGB')


class VGGFeatExtractor():
    """VGG16 backbone wrapper
    """
    def __init__(self, device):
        self.device = device
        self.url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(self.url) as f:
            self.module = torch.jit.load(f).eval().to(device)

    def __call__(self, img): # PIL
        img = self._preprocess(img, self.device)
        feat = self.module(img)
        return feat # (1, 1000)

    def _preprocess(self, img, device):
        img = img.resize((256,256), PIL.Image.LANCZOS)
        img = np.array(img, dtype=np.uint8)
        img = torch.tensor(img.transpose([2,0,1])).unsqueeze(dim=0)
        return img.to(device)


class Generator():
    """StyleGAN2 generator wrapper
    """
    def __init__(self, ckpt, device):
        with dnnlib.util.open_url(ckpt) as f:
            old_G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device)
        resolution = old_G.img_resolution
        generator_config = GENERATOR_CONFIGS(resolution=resolution)
        self.G_kwargs = generator_config.G_kwargs
        self.common_kwargs = generator_config.common_kwargs

        self.G = dnnlib.util.construct_class_by_name(**self.G_kwargs, **self.common_kwargs).eval().requires_grad_(False).to(device)
        copy_params_and_buffers(old_G, self.G, require_all=False)
        del old_G
        G = self.G

        self.style_layers = [
            f'G.synthesis.b{feat_size}.{layer}.affine'
            for feat_size in [pow(2,x) for x in range(2, int(np.log2(resolution))+1)]
            for layer in ['conv0', 'conv1', 'torgb']]
        del(self.style_layers[0])
        scope = locals()
        self.to_stylespace = {layer:eval(layer, scope) for layer in self.style_layers}
        w_idx_lst = generator_config.w_idx_lst
        assert len(self.style_layers) == len(w_idx_lst)
        self.to_w_idx = {self.style_layers[i]:w_idx_lst[i] for i in range(len(self.style_layers))}

    def mapping(self, z, truncation_psi=0.7, truncation_cutoff=None, skip_w_avg_update=False):
        '''random z -> latent w
        '''
        return self.G.mapping(
            z, 
            None,
            truncation_psi=truncation_psi, 
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update
        )

    def mapping_stylespace(self, latent):
        '''latent w -> style s
        resolution | w_idx | # conv | # torgb | indices
                 4 |     0 |      1 |       1 |     0-1
                 8 |     1 |      2 |       1 |     1-3
                16 |     3 |      2 |       1 |     3-5
                32 |     5 |      2 |       1 |     5-7
                64 |     7 |      2 |       1 |     7-9
               128 |     9 |      2 |       1 |    9-11
               256 |    11 |      2 |       1 |   11-13     # for  256 resolution
               512 |    13 |      2 |       1 |   13-15     # for  512 resolution
              1024 |    15 |      2 |       1 |   15-17     # for 1024 resolution
        '''
        styles = dict()
        for layer in self.style_layers:
            module = self.to_stylespace.get(layer)
            w_idx = self.to_w_idx.get(layer)
            styles[layer] = module(latent.unbind(dim=1)[w_idx])
        return styles

    def synthesis_from_stylespace(self, latent, styles):
        '''style s -> generated image
        modulated conv2d,  synthesis layer.weight,  noise
        forward after styles = affine(w)
        '''
        return self.G.synthesis(latent, styles=styles, noise_mode='const')

    def synthesis(self, latent):
        '''latent w -> generated image
        '''
        return self.G.synthesis(latent, noise_mode='const')


class e4eEncoder:
    '''e4e Encoder
    img paths -> latent w
    '''
    def __init__(self, device):
        self.device = device

    def __call__(self, target_pils):
        dataset = ImagesDataset(
            target_pils,
            self.device,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
        coach = MultiIDCoach(dataloader, device=self.device)
        latents = list()
        for fname, image in dataloader:
            latents.append(coach.get_e4e_inversion(image))
        latents = torch.cat(latents)
        return latents


class PivotTuning:
    '''pivot tuning inversion
    latent, style -> latent, style, 

    mode
    - 'latent' : use latent pivot
    - 'style' : use style pivot
    '''
    def __init__(self, device, G, mode='w'):
        assert mode in ['w', 's']
        self.device = device
        self.G = G
        self.mode = mode
        self.resolution = G.img_resolution

    def __call__(self, latent, target_pils):
        dataset = ImageLatentsDataset(
            target_pils,
            latent, 
            self.device,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])],),
            self.resolution,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        coach = MultiIDCoach(
            dataloader, 
            device=self.device, 
            generator=self.G,
            mode=self.mode
        )
        # run coach by self.mode
        new_G = coach.train_from_latent()
        return new_G
