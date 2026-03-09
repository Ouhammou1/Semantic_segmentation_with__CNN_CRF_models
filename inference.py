# # -*- coding: utf-8 -*-
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image
# from scipy.ndimage import gaussian_filter

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# VOC_CLASSES = [
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#     'diningtable', 'dog', 'horse', 'motorbike', 'person',
#     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]

# def get_palette():
#     palette = []
#     for i in range(256):
#         r = (i & 1) * 128 + ((i >> 3) & 1) * 16 + ((i >> 6) & 1) * 2
#         g = ((i >> 1) & 1) * 128 + ((i >> 4) & 1) * 16 + ((i >> 7) & 1) * 2
#         b = ((i >> 2) & 1) * 128 + ((i >> 5) & 1) * 16
#         palette += [r, g, b]
#     return palette


# def apply_crf(prob_map):
#     num_classes, h, w = prob_map.shape
#     refined = prob_map.copy()
#     for _ in range(5):
#         for c in range(num_classes):
#             refined[c] = gaussian_filter(refined[c], sigma=1.5)
#         refined = np.exp(refined - refined.max(axis=0, keepdims=True))
#         refined /= refined.sum(axis=0, keepdims=True) + 1e-8
#     return refined.argmax(axis=0)


# def load_model():
#     from train import SegmentationModel
#     model = SegmentationModel(num_classes=21).to(DEVICE)
#     model.load_state_dict(torch.load('models/segmentation_model.pth', map_location=DEVICE))
#     model.eval()
#     return model


# def run_inference(image_path, output_path, use_crf=True):
#     model = load_model()

#     image = Image.open(image_path).convert('RGB')
#     orig_size = image.size

#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     input_tensor = transform(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         output   = model(input_tensor)
#         prob_map = F.softmax(output, dim=1).squeeze(0).cpu().numpy()

#     if use_crf:
#         seg_map = apply_crf(prob_map)
#     else:
#         seg_map = prob_map.argmax(axis=0)

#     result = Image.fromarray(seg_map.astype(np.uint8), mode='P')
#     result.putpalette(get_palette())
#     result = result.convert('RGB').resize(orig_size)
#     result.save(output_path)

#     detected = [VOC_CLASSES[i] for i in np.unique(seg_map) if i < len(VOC_CLASSES)]
#     return output_path, detected
















    # -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_palette():
    palette = []
    for i in range(256):
        r = (i & 1) * 128 + ((i >> 3) & 1) * 16 + ((i >> 6) & 1) * 2
        g = ((i >> 1) & 1) * 128 + ((i >> 4) & 1) * 16 + ((i >> 7) & 1) * 2
        b = ((i >> 2) & 1) * 128 + ((i >> 5) & 1) * 16
        palette += [r, g, b]
    return palette


def apply_dense_crf(image_np, prob_map):
    """
    Real DenseCRF inference (Krahenbuhl & Koltun, 2011).
    Models pixel labeling as a fully-connected MRF.
    Inference via mean-field approximation - Bishop ss9.3

    image_np : HxWx3 uint8 RGB image
    prob_map : num_classes x H x W softmax probabilities from CNN
    """
    num_classes, h, w = prob_map.shape

    d = dcrf.DenseCRF2D(w, h, num_classes)

    # Unary potentials from CNN softmax output
    unary = unary_from_softmax(prob_map)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)

    # Smoothness kernel: Gaussian over position only
    # Penalizes label discontinuities between nearby pixels
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Appearance kernel: Gaussian over position AND color (bilateral)
    # Sharpens boundaries along color edges - key for segmentation quality
    img_c = np.ascontiguousarray(image_np)
    d.addPairwiseBilateral(
        sxy=80,    # spatial sigma
        srgb=13,   # color sigma
        rgbim=img_c,
        compat=10
    )

    # Mean-field inference: 5 iterations
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(h, w)


def load_model():
    from train import SegmentationModel
    model = SegmentationModel(num_classes=21).to(DEVICE)
    model.load_state_dict(torch.load(
        'models/segmentation_model.pth', map_location=DEVICE
    ))
    model.eval()
    return model


def run_inference(image_path, output_path, use_crf=True):
    model = load_model()

    image = Image.open(image_path).convert('RGB')
    orig_size = image.size

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output   = model(input_tensor)
        prob_map = F.softmax(output, dim=1).squeeze(0).cpu().numpy()

    if use_crf:
        img_resized = np.array(image.resize((224, 224)), dtype=np.uint8)
        seg_map = apply_dense_crf(img_resized, prob_map)
    else:
        seg_map = prob_map.argmax(axis=0)

    result = Image.fromarray(seg_map.astype(np.uint8), mode='P')
    result.putpalette(get_palette())
    result = result.convert('RGB').resize(orig_size)
    result.save(output_path)

    detected = [
        VOC_CLASSES[i]
        for i in np.unique(seg_map)
        if i > 0 and i < len(VOC_CLASSES)
    ]

    return output_path, detected