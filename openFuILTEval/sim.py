import torch

from openFuILTEval.utils import SimConfig, PrintConfig
from pylitho import Abbe

class SimEval:
    def __init__(self, 
                 mask : torch.Tensor,
                 print_config : PrintConfig,
                 sim_config : SimConfig,
                 ) -> None:
        
        self.mask = mask
        self.print_config  : PrintConfig = print_config
        self.sim_config : SimConfig = sim_config   
        assert len(mask.shape) == 2, f"Mask need to be 2D tensor, but we got {len(mask.shape)}"
        assert mask.shape[0] == mask.shape[1], f"Mask need to be square, but we got {mask.shape[0]} and {mask.shape[1]}"
    
        
    @torch.no_grad()
    def sim(self):
        aerial_image : torch.Tensor = None
        model = Abbe(self.mask.shape[0] * self.sim_config.pixel,
                           self.sim_config.pixel,
                           self.sim_config.sigma,
                           self.sim_config.NA,
                           self.sim_config.wavelength, 
                           self.sim_config.defocus
                           )
        aerial_image = model(self.mask)
        assert(aerial_image is not None)
        if aerial_image.dim() == 4:
            aerial_image = aerial_image.squeeze(0)
        assert aerial_image.dim() == 3 and aerial_image.shape[0] == 3, \
            f"Aerial image should be a 3D tensor with 3 channels, but we got {aerial_image.shape}"
        aerial_image.sub_(self.print_config.targetIntensity)  
        aerial_image.mul_(self.print_config.stepness)       
        torch.sigmoid(aerial_image, out=aerial_image)  
        torch.round(aerial_image. out=aerial_image)
        return aerial_image
    
    
