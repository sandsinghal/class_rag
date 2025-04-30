import torch

# Load the model.bin
state_dict = torch.load('./outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/e5_small_v2/flan_t5_xl/epoch/5/2025_04_27/16_33_24/train/pytorch_model.bin', map_location='cpu')

# Print each parameter name and its shape
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
