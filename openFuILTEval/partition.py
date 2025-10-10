from typing import List, Tuple
from openFuILT.src.geometry import BBox, Point
from openFuILT.src.partition import Tile

from openFuILTEval.utils import scalePadding

class SimPartition:
    
    def __init__(self,
                pixel : int,
                bbox : BBox,
                macro_size : List[int],
                overlap_rate : float = 0.2) -> None:
        
        self.pixel : int = pixel
        self.macro_size : List[int] = macro_size
        
        pre_padding = scalePadding([bbox.getWidth(), bbox.getHeight()], self.macro_size)

        # scale to dividable size
        self.bbox : BBox = BBox(
            bbox.getLowLeft() - [pre_padding, pre_padding],
            bbox.getUpRight() + [pre_padding, pre_padding]
        )
    
        # the height and width of pure region
        self.tile_width, self.tile_height = None, None        
        self.target_width, self.target_height = None, None
        
        self.tile_list : List[Tile] = []
        self.overlap_rate = overlap_rate
        
        # overall bbox after adding overlap
        self.bbox_virtual : BBox = None
        
    def __compute_tile_size(self):
        bb_width = self.bbox.getWidth()
        assert bb_width % self.macro_size[0] == 0, f"The total mask width {bb_width} need to be divide by {self.macro_size[0]}"
        self.tile_width = bb_width // self.macro_size[0]

        bb_height = self.bbox.getHeight()
        assert bb_height % self.macro_size[1] == 0, f"The total mask width {bb_height} need to be divide by {self.macro_size[1]}"
        self.tile_height = bb_height // self.macro_size[1]
        
        assert self.tile_width == self.tile_height, f"Tile width and height need to be the same, but we got {self.tile_width} and {self.tile_height}"
        
        self.target_width = self.tile_width * self.pixel
        self.target_height = self.tile_height * self.pixel
    
    def __generate_tiles_index(self, i, j):
        assert i < self.macro_size[0] and j < self.macro_size[1]
        bb_start = self.bbox.getLowLeft()
        ll = Point(bb_start.x() + i * self.tile_width, bb_start.y() + j * self.tile_height)
        ur = Point(ll.x() + self.tile_width, ll.y() + self.tile_height)
        assert self.bbox.isInside(ll) and self.bbox.isInside(ur), f"{ll} and {ur} need to inside {self.bbox}"
        bbox = BBox(ll, ur)        
        return Tile(bbox)
    
    def __compute_overlaping_region(self):
        from math import ceil
        w_ol = ceil(self.tile_width * self.overlap_rate)
        h_ol = ceil(self.tile_height * self.overlap_rate)
        assert w_ol == h_ol
        return [w_ol, h_ol]
    
    def __generate_overlaps(self):
        w_ol, h_ol = self.__compute_overlaping_region()
        for i in range(self.macro_size[0]):
            for j in range(self.macro_size[1]):
                tile = self.__generate_tiles_index(i, j)
                tile.setIndex(i, j)
                tile.bbox_ol = BBox(tile.bbox.getLowLeft() + [-w_ol, -h_ol], tile.bbox.getUpRight() + [w_ol, h_ol])
                self.tile_list.append(tile)
    
    def __generate_virtual_box(self):
        x1, y1, x2, y2 = None, None, None, None
        for tile in self.tile_list:
            bb : List[int] = tile.bbox_ol.list()
            if x1 == None or y1 == None:
                x1, y1, x2, y2 = bb
            else:
                x1 = min(x1, bb[0])
                y1 = min(y1, bb[1])
                x2 = max(x2, bb[2])
                y2 = max(y2, bb[3])
        self.bbox_virtual = BBox(Point(x1, y1), Point(x2, y2))
                
                
    def __check(self):
        import numpy as np
        W_ol, H_ol = None, None
        bboxes = np.zeros(4)
        for tile in self.getTiles():
            if W_ol == None:
                W_ol, H_ol = tile.bbox_ol.getWidth(), tile.bbox_ol.getHeight()
            else:
                assert W_ol, H_ol == (tile.bbox_ol.getWidth(), tile.bbox_ol.getHeight())
            bboxes = np.vstack([bboxes, tile.getBBoxOverlap(numpy=True)])
        bboxes = bboxes[1:, :]
        np.isclose(bboxes.min(axis=0)[0:2], self.bbox.getLowLeft().numpy())
        np.isclose(bboxes.max(axis=0)[2:], self.bbox.getUpLeft().numpy())
                
    def partition(self):
        self.__compute_tile_size()
        self.__generate_overlaps()
        self.__generate_virtual_box()
        
        self.__check()

    def getTiles(self) -> List[Tile]:
        return self.tile_list
    
    def getBBoxVirtual(self) -> BBox:
        if self.bbox_virtual is None:
            raise ValueError("You need to call partition() before getBBoxVirtual()")
        return self.bbox_virtual
    