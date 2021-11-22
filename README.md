# StyleCLIP-PyTorch: Text-Driven Manipulation of StyleGAN Imagery 
- With PTI (Pivot Tuning Inversion)
- Global Direction Methods

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
- Docker build
`$ sh build_img.sh`
`$ sh build_container.sh [container-name]`

- Install package
`$ docker start [container-name]`
`$ docker attach [container-name]`
`$ pip install -v -e .`

# Pretrained weights
- [dlib landmarks detector](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing) 
- [FFHQ e4e encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing)
- [FFHQ stylegan2-ada](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) 

Download and save this pretrained weights in `pretrained/` directory

# Overall Flow
1. Input image face align
2. Project image to latent space(W projector or e4e encoder)
3. Convert latent to style
4. Input text prompt
    - Neutral / target text
5. Get delta t from input text, using clip text encoder
6. (Preprocessed) Get extracted global direction
7. Get delta s from global direction and delta t, that satisfy beta threshold
8. Manipulate style codes in style space
9. Visualize

# Manipulation option
- Source image
    - Input image projection
    - Generate z from random seed
- Text description(neutral, target)
- Manipulation strength (alpha)
- Disentangle threshold (beta) 

# TODO
- Cleanup configuration system -> need to test
- Cleanup e4e encoder wrapper -> use e4e encoder module directly
- Global direction module refactoring(especially in gpu usage)
