
import torch

from models.pfld import PFLDInference
from torch.utils.mobile_optimizer import optimize_for_mobile

checkpoint = torch.load('./checkpoint/snapshot/checkpoint.pth.tar', map_location=torch.device('cpu'))

pfld_backbone = PFLDInference()
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
pfld_backbone.eval()

scripted_model = torch.jit.script(pfld_backbone)

optimized_traced_model = optimize_for_mobile(scripted_model)
optimized_traced_model._save_for_lite_interpreter('./mobile/pfld_mobile.pt')
