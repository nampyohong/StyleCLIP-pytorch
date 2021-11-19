import clip
import numpy as np
import torch


def zeroshot_classifier(classnames, templates, model):
    device = model.token_embedding.weight.device
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts) #tokenize
            texts = texts.to(device)
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def get_delta_t(
        classnames, 
        model, 
        prompts=np.load('pretrained/imagenet_templates.npy').tolist()
    ):
    text_features = zeroshot_classifier(classnames, prompts, model).t()

    delta_t = (text_features[0] - text_features[1]).cpu().numpy()
    delta_t = delta_t/np.linalg.norm(delta_t)
    return delta_t


if __name__ == "__main__":
    device = torch.device('cuda:3')
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    classnames=['face', 'face with glasses']

    delta_t = get_delta_t(classnames, model)
    print(delta_t[:10])
    delta_t = get_delta_t(classnames, model, ['a photo of {}'])
    print(delta_t[:10])
    print(delta_t.__class__)
    print('shape: ', delta_t.shape)
