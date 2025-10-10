import torch
from torch import nn

from queue import Queue
from threading import Thread
from typing import List, Tuple

from openFuILT.src.geometry import BBox, Point

from openFuILTEval.utils import SimConfig
from openFuILTEval.utils import PrintConfig
from openFuILTEval.utils import getMaskFromDisc
from openFuILTEval.utils import getScaledPaddingSize

from openFuILTEval.layout import TargetLayout

from openFuILTEval.partition import SimPartition

from openFuILTEval.metric import EPEChecker, getLoss, getPVBand, ShotCounter
from openFuILTEval.metric import LocalMetric, Metric

from openFuILTEval.sim import SimEval

from tqdm import tqdm

class Evaluator:
    def __init__(self,
                 pixel : int,
                 macro_size : List[int],
                 mask_path : str,
                 target_path : str,
                 overlap_rate : float = 0.2,
                 **kwargs
                 ) -> None:
        
        if kwargs.get("simConfig", None) is not None:
            self.simConfig : SimConfig = kwargs.get("simConfig")
        else:   
            self.simConfig : SimConfig = SimConfig(pixel = pixel)
        
        if kwargs.get("printConfig", None) is not None:
            self.printConfig  : PrintConfig = kwargs.get("printConfig")
        else:
            self.printConfig  : PrintConfig = PrintConfig()
        
        self.mask_path = mask_path
        self.target_path = target_path
        
        
        assert len(macro_size) == 2 and macro_size[0] == macro_size[1], \
            f"Macro size should be a list of two identical integers, but we got {macro_size}"
        self.macro_size = macro_size
        self.overlap_rate = overlap_rate
        
        
        self.mask : torch.Tensor = getMaskFromDisc(mask_path=self.mask_path,
                                                   is_torch=True).to(torch.float32)
        self.mask_bb = BBox(Point(0, 0), Point(self.mask.shape[0], self.mask.shape[1]))

        assert isinstance(self.mask_bb, BBox) and self.mask_bb.getHeight() == self.mask_bb.getWidth(), \
            f"Mask should be a square BBox, but we got {self.mask_bb}"

        self.layout = TargetLayout(
            pixel=self.simConfig.pixel,
            filePath=self.target_path
        )
        
        self.local_metric_result = Metric()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if kwargs.get("device_id", None) is not None:
            self.device = torch.device(f"cuda:{kwargs.get('device_id')}")

        
        
    def _maskPadding(self, mask : torch.Tensor, padding_config : Tuple[int]):
        padding = nn.ZeroPad2d(padding_config)
        return padding(mask)    
    
    
    def _eval(self, mask_tile : torch.Tensor, print_image : torch.Tensor, target_image : torch.Tensor):
        if target_image.sum().item() == 0:
            return LocalMetric(L2=0, EPE=0, PVBand=0)
        EPE = EPEChecker.check(print_image[0], target_image)
        L2 = getLoss(print_image[0], target_image)
        pvband = getPVBand(print_image[1], print_image[2])
        shot = ShotCounter.count(mask_tile)
        return LocalMetric(L2=L2, EPE=EPE, PVBand=pvband, SHOT=shot)
    
    
    def _drop(self, mask_tile : torch.Tensor, print_image : torch.Tensor, target_image : torch.Tensor):
        del target_image
        del print_image
        del mask_tile
        
        if self.device.type != "cpu":
            torch.cuda.empty_cache()
    
        
    def report(self):
        self.local_metric_result.report()
    
        
    def evaluate(self):
        
        partitioner = SimPartition(
            pixel=self.simConfig.pixel,
            bbox=self.mask_bb,
            macro_size=self.macro_size,
            overlap_rate=self.overlap_rate
        )
        
        partitioner.partition()
        
        padding_config = getScaledPaddingSize(self.mask_bb, partitioner.getBBoxVirtual())
        mask = self._maskPadding(self.mask, padding_config=padding_config)
    
        source_point = partitioner.bbox_virtual.getLowLeft()

        for tile in tqdm(partitioner.getTiles(), desc="Simulating and Evaluating"):
            bb_ol = tile.getBBoxOverlap(numpy=True)
            bb_target_ol = bb_ol * self.simConfig.pixel
            
            index = bb_ol.reshape((2, 2)) - source_point.numpy()
            index = index.reshape(-1).astype(int)
        
            mask_tile = mask[index[0]:index[2], index[1]:index[3]]
            target = self.layout.clipping(bb_target_ol.tolist())
            
            if self.device.type != "cpu":
                mask_tile = mask_tile.to(self.device)
                target = target.to(self.device)
            
            sim = SimEval(mask_tile, self.printConfig, self.simConfig)
            print_image : torch.Tensor = sim.sim() 

            assert print_image.shape[-2] == target.shape[-2] and print_image.shape[-1] == target.shape[-1]

            idx_in_ol = tile.getOriginIndexInBBoxOverlap(numpy=True) * self.simConfig.pixel
            print_image = print_image[:, idx_in_ol[0]:idx_in_ol[2], idx_in_ol[1]:idx_in_ol[3]]
            target = target[idx_in_ol[0]:idx_in_ol[2], idx_in_ol[1]:idx_in_ol[3]]
            
            metric = self._eval(mask_tile, print_image, target)
            self.local_metric_result.addTile(tile, metric)
            
            self._drop(mask_tile, print_image, target)
            