import time

import dlib
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy
from dlib_utils.face_alignment import image_align
from dlib_utils.landmarks_detector import LandmarksDetector
from torch_utils.misc import copy_params_and_buffers


class FaceLandmarksDetector:
    """Dlib landmarks detector wrapper
    """
    def __init__(self, model_path, tmpdir):
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
        return PIL.Image.open(self.tmp_align)


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
        self.G_kwargs = {
            'class_name': 'training.networks.Generator',
            'z_dim': 512,
            'w_dim': 512,
            'mapping_kwargs': {'num_layers': 8},
            'synthesis_kwargs': {
                'channel_base': 32768,
                'channel_max': 512,
                'num_fp16_res': 4,
                'conv_clamp': 256
            }
        }
        self.common_kwargs = {'c_dim': 0, 'img_resolution': 1024, 'img_channels': 3}

        with dnnlib.util.open_url(ckpt) as f:
            old_G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device)

        self.G = dnnlib.util.construct_class_by_name(**self.G_kwargs, **self.common_kwargs).eval().requires_grad_(False).to(device)
        copy_params_and_buffers(old_G, self.G, require_all=False)
        del old_G
        G = self.G

        self.style_layers = [
            f'G.synthesis.b{feat_size}.{layer}.affine'
            for feat_size in [pow(2,x) for x in range(2, 11)]
            for layer in ['conv0', 'conv1', 'torgb']]
        del(self.style_layers[0])
        scope = locals()
        self.to_stylespace = {layer:eval(layer, scope) for layer in self.style_layers}
        w_idx_lst = [0,1,1,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17]
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
               256 |    11 |      2 |       1 |   11-13 
               512 |    13 |      2 |       1 |   13-15
              1024 |    15 |      2 |       1 |   15-17
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

        # TODO : latent 제거
        '''
        return self.G.synthesis(latent, styles=styles, noise_mode='const')

    def synthesis(self, latent):
        '''latent w -> generated image
        '''
        return self.G.synthesis(latent, noise_mode='const')
