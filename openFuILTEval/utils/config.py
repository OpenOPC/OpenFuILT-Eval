from typing import List, Union


class SimConfig:
    def __init__(self,
                pixel : int, 
                sigma=0.05, 
                NA=1.35, 
                wavelength=193,
                defocus : Union[None, List[int]] = [0, 30, 60]) -> None:
        self.pixel = pixel
        self.sigma = sigma
        self.NA = NA
        self.wavelength = wavelength
        self.defocus = defocus
        
        
class PrintConfig:
    
    def __init__(self,
                 stepness=50,
                 targetIntensity=0.225) -> None:
        self.stepness = stepness
        self.targetIntensity = targetIntensity