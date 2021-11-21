StyleCLIP-PyTorch: Text-Driven Manipulation of StyleGAN Imagery 
+ PTI (Pivot Tuning Inversion)

# Following will be updated soon
- Manipulator for 256x256, 512x512 resolution
- Explanation and instruction for module
- Colab notebook demo

--------------
References

1. stylegan2-ada-pytorch
- https://github.com/NVlabs/stylegan2-ada-pytorch

2. CLIP
- https://github.com/openai/CLIP.git

3. StyleCLIP(tensorflow, ORIGINAL REPO)
- https://github.com/orpatashnik/StyleCLIP

4. Pivot Tuning Inversion
- https://github.com/danielroich/PTI


--------------
Invert Image
- W space optimization
- W+ space optimization
- Pivot tuning inversion (TODO)
    - Tuning after style space

Style Extractor
- from latent W/W+ space to S/S+ space
- generate synth image from style space

Manipulator
- Global style direction extractor
- Style manipulator

Get style direction from CLIP text space
- Text Direction Extractor
    - prompt engineering for face manipulation
- mapper for combining multimodal style by beta threshold

Test Pipeline (TODO)


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
