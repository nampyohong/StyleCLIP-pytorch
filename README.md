StyleCLIP-PyTorch: Text-Driven Manipulation of StyleGAN Imagery 
+ PTI (Pivot Tuning Inversion)


stylegan2-ada-pytorch
- https://github.com/NVlabs/stylegan2-ada-pytorch

StyleCLIP(tensorflow)
- https://github.com/orpatashnik/StyleCLIP

Pivot Tuning Inversion
- https://github.com/danielroich/PTI


Image Input : PIL, CLIP image encoder
Text Input : CLIP text encoder

Image Invert
- W space optimization
- W+ space optimization
- Pivot tuning inversion
    - Style based generator tuning

Style Extractor
- mapper from latent to style
- from latent W/W+ space to S/S+ space
- generate synth image from style space

Direction Extractor (pytorch)
- Style Direction Extractor
    - SingleChannel.py in original tf repo
    - implement dataset
    - implement dataloader
- Text Direction Extractor
    - apply prompt engineering for face manipulation

Test Pipeline



--------------
Flow
- Extract W latent, S style, S mean/std from 100000 / 2000 / 100000 
  generated Z random latent
- Extract global direction from style space

- Align target face image using Dlib face landmarks detector
- Get projected W/W+ tensor from aligned image
- Pivot Tuning after style space(only for modulated conv2d layer)


- manipulation strength (alpha)
- disentangle threshold (beta) 
