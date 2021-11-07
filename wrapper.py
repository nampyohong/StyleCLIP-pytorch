import time

import dlib
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy
from dlib_utils.face_alignment import image_align
from dlib_utils.landmarks_detector import LandmarksDetector


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
        with dnnlib.util.open_url(ckptpath) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device)

    def __call__(self, latent):
        synth_img = self.G.synthesis(latent, noise_mode='const')
        synth_img = (synth_img + 1) * (255/2)
        synth_img = synth_img.permute(0,2,3,1).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
        return synth_img
