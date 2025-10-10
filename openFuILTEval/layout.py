import gdspy
from gdspy import Cell
from typing import List
import numpy as np
import torch

from openFuILT.src.geometry import RTree
from openFuILT.src.utils.functions import polygon_reduction
from openFuILT.src.geometry import PolygonClipper
from openFuILT.src.geometry import rectanglize, to_image

def _to_image(box : np.ndarray, polygon : List[np.ndarray]) -> torch.Tensor:
    w, h = int(box[2] - box[0]), int(box[3] - box[1])
    rects = []
    for p in polygon:
        p_s = p - np.array([box[0], box[1]])
        rects += rectanglize(p_s.astype(np.int32))
    img = to_image([w, h], rects)
    return img

class TargetLayout:
    nano_unit = 1e-9
    def __init__(self,
                filePath : str,
                pixel : int,
                unit : float = 1e-6, 
                layer=0) -> None:
        self.unit = unit
        self.layer = layer
        
        self.pixel = pixel
        self.filePath = filePath
        
        self.gdslib = gdspy.GdsLibrary(unit=unit)
        self.gdslib.read_gds(self.filePath)
        self.top_cell : Cell = self.gdslib.top_level()[0]
        self.scales = self.unit / self.nano_unit

        self.polygons : List[np.ndarray] = []
        self._gen_polygon_tensor()
        
        self._build_rtree()
        
    def _gen_polygon_tensor(self):
        for p in self.top_cell.get_polygons():
            p_scaled = p * self.scales # shape in [N, 2] 
            p_scaled = p_scaled
            self.polygons.append(p_scaled)
            
    def get_polygons(self) -> List[np.ndarray]:
        return self.polygons
    
    def _build_rtree(self):
        self.rtree = RTree()
        for idp, p in enumerate(self.polygons):
            p_bb = polygon_reduction(p).tolist()
            self.rtree.insert(idp, p_bb, p)
    
    def clipping(self, bbox : List[int]) -> torch.Tensor:
          
            polygons_in_tile = self.rtree.intersection(bbox) 
            
            # counter-clockwise
            clip_bbox = np.array([
                [bbox[0],bbox[1]],
                [bbox[2],bbox[1]],
                [bbox[2],bbox[3]],
                [bbox[0],bbox[3]],
            ])
            
            polygons_clipped = []
            for poly in polygons_in_tile:
                result = PolygonClipper()(poly, clip_bbox)
                if result is not None:
                    polygons_clipped.append(result)
                    
            target = _to_image(bbox, polygons_clipped)
            return target
            
           