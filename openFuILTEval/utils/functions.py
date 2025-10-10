from typing import List, Union, Tuple
import numpy as np
from openFuILT.src.geometry import BBox, Point
import pickle
import torch

def scalePadding(mask_shape : List[int], size : List[int]) -> int:
    H, W = mask_shape
    padding = 0
    while H % size[0] != 0 or W % size[1] != 0:
        padding += 1
        H, W = H + 2, W + 2
    return padding


def getScaledPaddingSize(mask_bbox : BBox, scaled_bbox : BBox) -> Tuple[int]:
    mask_width = mask_bbox.getWidth()
    mask_height = mask_bbox.getHeight()
    scaled_width = scaled_bbox.getWidth()
    scaled_height = scaled_bbox.getHeight()

    assert (scaled_width - mask_width) % 2 == 0
    assert (scaled_height - mask_height) % 2 == 0

    pad_x = (scaled_width - mask_width) // 2
    pad_y = (scaled_height - mask_height) // 2

    return (pad_x, pad_x, pad_y, pad_y)

def getMaskFromDisc(mask_path : str,
                    is_torch : bool=True):
    
    import torch
    with open(mask_path, 'rb') as file:
        mask = pickle.load(file)
    file.close()
    assert(isinstance(mask, np.ndarray))
    return mask if not is_torch else torch.from_numpy(mask)
