# StyleCLIP-PyTorch: Text-Driven Manipulation of StyleGAN Imagery 
with PTI (Pivot Tuning Inversion)

# Following will be updated soon
- Manipulator for 256x256, 512x512 resolution, supports other datasets
- Explanation and instruction for module
- Colab notebook demo

# References
1. [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
2. [CLIP](https://github.com/openai/CLIP.git)
3. [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
4. [Pivot Tuning Inversion](https://github.com/danielroich/PTI)

# Installation
- docker build
$ sh build_img.sh
$ sh build_container.sh [container_name]
 
- install package
$ docker start [container_name]
$ docker attach [container_name]
$ pip install -v -e .

# Pretrained weights
- [dlib landmarks detector](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing) 
- [FFHQ e4e encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing)
- [FFHQ stylegan2-ada](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) 

Download and save this pretrained weights in `pretrained/` directory

# Overall Flow
1. input image face align
2. project image to latent space(W projector or e4e encoder)
3. convert latent to style
4. input text prompt
    - neutral / target text
5. get delta t from input text, using clip text encoder
6. (preprocessed) get extracted global direction
7. get delta s from global direction and delta t, that satisfy beta threshold
8. manipulate style codes in style space
9. visualize

# manipulation option
- source image
    - input image projection
    - generate z from random seed
- text description(neutral, target)
- manipulation strength (alpha)
- disentangle threshold (beta) 

# TODO
- Global direction module refactoring(especially in gpu usage)
- Cleanup configuration system
