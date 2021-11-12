StyleCLIP-PyTorch: Text-Driven Manipulation of StyleGAN Imagery 
+ PTI (Pivot Tuning Inversion)


stylegan2-ada-pytorch
- https://github.com/NVlabs/stylegan2-ada-pytorch

CLIP
- https://github.com/openai/CLIP.git

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
Overall Flow
1. input image face align
2. project image to latent space(W/W+)
3. convert latent to style

4. input text prompt
    - neutral / target text
5. get delta t from input text, using clip text encoder

6. (preprocessed) get extracted global direction
7. get delta s from global direction and delta t, that satisfy beta threshold

8. manipulate style codes in style space
9. visualize


--------------
manipulation option
- input image or generate z from random seed
- text description(neutral, target)
- manipulation strength (alpha)
- disentangle threshold (beta) 
