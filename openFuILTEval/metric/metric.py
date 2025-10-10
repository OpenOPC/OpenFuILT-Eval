from typing import NamedTuple, Dict
import cv2
import torch
import numpy as np
import torch.nn.functional as func 

from openFuILT.src.utils import get_logger

logger = get_logger("OpenFuILTEval")

from openFuILT.src.partition import Tile

_has_adabox = True
try:
    from adabox import proc, tools
except ImportError:
    _has_adabox = False
    logger.warning("adabox is not installed, ShotCounter will not work.")


def getLoss(image : torch.tensor, target : torch.tensor):
    assert image.shape == target.shape
    return torch.sum((image - target) ** 2).item()

def getPVBand(positive : torch.tensor, negetive : torch.tensor):
    return getLoss(positive, negetive)

class ShotCounter: 
    @staticmethod
    def count(mask):
        if not _has_adabox: 
            return None
        if not isinstance(mask, torch.Tensor): 
            mask = torch.tensor(mask, dtype=torch.float, device=torch.cpu())
        image = image.detach().cpu().numpy().astype(np.uint8)
        comps, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        rectangles = []
        for label in range(1, comps): 
            pixels = []
            for idx in range(labels.shape[0]): 
                for jdx in range(labels.shape[1]): 
                    if labels[idx, jdx] == label: 
                        pixels.append([idx, jdx, 0])
            pixels = np.array(pixels)
            x_data = np.unique(np.sort(pixels[:, 0]))
            y_data = np.unique(np.sort(pixels[:, 1]))
            if x_data.shape[0] == 1 or y_data.shape[0] == 1: 
                rectangles.append(tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max()))
                continue
            (rects, sep) = proc.decompose(pixels, 4)
            rectangles.extend(rects)
        return len(rectangles)

class LocalMetric(NamedTuple):
    EPE : int = None
    L2 : float = None
    SHOT : int = None
    PVBand : float = None
    
    
class Metric:
    def __init__(self) -> None:
        self.tile_metric_map : Dict[Tile, LocalMetric] = dict()
        self.EPE, self.L2, self.SHOT, self.PVBand = [0, 0, 0, 0]
        self.visited = set()
    
    def addTile(self, tile : Tile, metric : LocalMetric):
        self.tile_metric_map[tile] = metric
        if metric.EPE is not None:
            self.EPE += metric.EPE
        if metric.L2 is not None:
            self.L2 += metric.L2
        if metric.SHOT is not None:
            self.SHOT += metric.SHOT
        if metric.PVBand is not None:
            self.PVBand += metric.PVBand
        self.visited.add(tile)
        
    def report(self):
        assert(len(self.visited) == len(self.tile_metric_map))
        logger.info(f"[EPE]: {self.EPE}, [L2]: {self.L2}, [SHOT]: {self.SHOT}, [PVBand]: {self.PVBand}")