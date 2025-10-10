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
        ariel_image : torch.Tensor = None
        model = Abbe(self.mask.shape[0] * self.sim_config.pixel,
                           self.sim_config.pixel,
                           self.sim_config.sigma,
                           self.sim_config.NA,
                           self.sim_config.wavelength, 
                           self.sim_config.defocus
                           )
        ariel_image = model(self.mask)
        assert(ariel_image is not None)
        if ariel_image.dim() == 4:
            ariel_image = ariel_image.squeeze(0)
        assert ariel_image.dim() == 3 and ariel_image.shape[0] == 3, \
            f"Ariel image should be a 3D tensor with 3 channels, but we got {ariel_image.shape}"
        print_image = torch.sigmoid(self.print_config.stepness * (ariel_image - self.print_config.targetIntensity))
        return print_image
    
    