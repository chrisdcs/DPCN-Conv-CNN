import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

import torch
import torchvision.datasets as datasets
import kornia as K


from torchvision import utils
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid


from model.Layer1 import Layer1
from model.Layer2 import Layer2
from model.Layer3 import Layer3

from utils.data import ZCA_Loader
import os
from pathlib import Path

def tensor_to_pil(image_tensor):
    return T.ToPILImage()(image_tensor)

# Normalize whitened images into the range [0, 255]
def normalize_to_uint8(image_tensor):
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    
    # Apply min-max normalization
    normalized = (image_tensor - min_val) / (max_val - min_val) * 255.0
    
    # Convert to uint8
    return normalized.byte()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = "runs/14/"
data_type = 'val'
data_dir = f"data/tiny-imagenet-200/{data_type}"
Dx = 64
m11 = 768
m12 = 768
m21 = 768
m22 = 768
m31 = 768
m32 = 768

zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
state_dict = torch.load('data/tiny-imagenet-200/zca_model.pt')
zca.transform_matrix = state_dict['transform_matrix']
zca.transform_inv = state_dict['transform_inv']
zca.mean_vector = state_dict['mean_vector']
zca.fitted = state_dict['fitted']
zca = zca.to(device)

layer1 = Layer1(32, 0, 1, R_lr=0, lmda=0, device=device, save_dir='') # ignore those parameters
layer2 = Layer2(32, 0, 1, R_lr=0, lmda=0, device=device, save_dir='') # ignore those parameters
layer3 = Layer3(32, 0, 1, R_lr=0, lmda=0, device=device, save_dir='') # ignore those parameters

layer1.load_state_dict(torch.load(folder + "layer1_1_f9_01.pt"))
layer2.load_state_dict(torch.load(folder + "layer1_2_f9_01.pt"))
layer3.load_state_dict(torch.load(folder + "layer1_3_f9_01.pt"))

transform_tiny = T.Compose([
    T.ToTensor(),  # Convert PIL image to tensor
])

tiny_imgnet = datasets.ImageFolder(
        f"data/tiny-imagenet-200/{data_type}",
        transform = transform_tiny)
batch_size = 32
tiny_loader = torch.utils.data.DataLoader(tiny_imgnet, batch_size=batch_size, shuffle=False)
save_dir_u2 = f"data/tiny-imagenet-200-DPCN-u2/{data_type}"
save_dir_u3 = f"data/tiny-imagenet-200-DPCN-u3/{data_type}"
if not os.path.exists(save_dir_u2):
    os.makedirs(save_dir_u2)

if not os.path.exists(save_dir_u3):
    os.makedirs(save_dir_u3)

for batch_idx, (images, labels) in enumerate(tiny_loader):
    
    if batch_idx % 100 == 0:
        print("Progress: {:.2f}%".format(100 * batch_idx / len(tiny_loader)))
    if batch_idx < 624:
        continue
    with torch.no_grad():
        img_batch = zca(images)
    img_batch = img_batch.to(device)
    labels = labels.to(device)
    
    for t in range(2):
        # obtain states in the step t-1
        if t == 0:
            Rt_1_1 = torch.zeros((batch_size, m11, Dx, Dx), device=device)
            Rt_1_2 = torch.zeros((batch_size, m21, Dx//2, Dx//2), device=device)
            Rt_1_3 = torch.zeros((batch_size, m31, Dx//4, Dx//4), device=device)
        else:
            Rt_1_1 = layer1.R.data.clone().detach()
            Rt_1_2 = layer2.R.data.clone().detach()
            Rt_1_3 = layer3.R.data.clone().detach()
        
        Td_1 = (layer2.U(Rt_1_2)).clone().detach() #predicted cause of layer1 based on the states of layer2
        Td_2 = (layer3.U(Rt_1_3)).clone().detach() #predicted cause of layer2 based on the states of layer3
        Td_3 = layer3.u.data.clone().detach()   #predicted cause of layer3, that is the cause at previous step 
        
        # infer states and causes using Fista
        
        if t  == 0:
            #print("inference start")
            x1, u1, xp1 = layer1.Fista_(img_batch, Rt_1_1, Td_1 , t, )
            #print("inference done")
            img4layer2 = layer1.u.data.clone().detach()
            
            img4layer2_old = layer1.u.data.clone().detach()

            x2, u2, xp2 = layer2.Fista_(img4layer2, Rt_1_2, Td_2, t, )
            img4layer3 = layer2.u.data.clone().detach()
            
            img4layer3_old = layer2.u.data.clone().detach()
            x3, u3, xp3 = layer3.Fista_(img4layer3, Rt_1_3, Td_3, t, )
        else:
            #print("inference start")
            x12, u12, xp12 = layer1.Fista_(img_batch, Rt_1_1, Td_1 , t, )
            #print("inference done")
            img4layer2 = layer1.u.data.clone().detach()

            x22, u22, xp22 = layer2.Fista_(img4layer2, Rt_1_2, Td_2, t, )
            img4layer3 = layer2.u.data.clone().detach()
            x32, u32, xp32 = layer3.Fista_(img4layer3, Rt_1_3, Td_3, t, )
            """
            x32 = x3
            u32 = u3
            xp32 = xp3
            
            Rt_1_2 = layer2.R.data.clone().detach()
            Td_2 = (layer3.U(Rt_1_3)).clone().detach()
            x22, u22, xp22 = layer2.Fista_(img4layer2_old, Rt_1_2, Td_2, t, )
            img4layer3 = layer2.u.data.clone().detach()
            
            Rt_1_1 = layer1.R.data.clone().detach()
            Td_1 = (layer2.U(Rt_1_2)).clone().detach()
            x12, u12, xp12 = layer1.Fista_(img_batch, Rt_1_1, Td_1 , t, )
            img4layer2 = layer1.u.data.clone().detach()
            #Rt_1_3 = layer3.R.data.clone().detach()
            """
            
            #x1 = torch.cat((x1,x12),0)
            #u1 = torch.cat((u1,u12),0)
            #xp1 = torch.cat((xp1,xp12),0)
            
            #x2 = torch.cat((x2,x22),0)
            #u2 = torch.cat((u2,u22),0)
            #xp2 = torch.cat((xp2,xp22),0)
            
            #x3 = torch.cat((x3,x32),0)
            #u3 = torch.cat((u3,u32),0)
            #xp3 = torch.cat((xp3,xp32),0)
            
            #Rt_1_1 = torch.cat((torch.zeros((batch_size, m11, Dx, Dx), device=device),Rt_1_1),0)
            #Rt_1_2 = torch.cat((torch.zeros((batch_size, m21, Dx//2, Dx//2), device=device),Rt_1_2),0)
            #Rt_1_3 = torch.cat((torch.zeros((batch_size, m31, Dx//4, Dx//4), device=device),Rt_1_3),0)
            
            #img_batch = torch.cat((img_batch,img_batch),0)
            #img4layer2 = torch.cat((img4layer2_old,img4layer2),0)
            #img4layer3 = torch.cat((img4layer3_old,img4layer3),0)
            
            #recon_11 = layer1.U(x12)
            #recon_12 = layer1.U(layer1.unpool(layer1.B(u12),layer1.index))
            #recon_21 = layer1.U(layer1.unpool(layer1.B(layer2.U(x22)),layer1.index))#layer1.U(layer1.B(layer2.U(x22)))
            recon_22 = layer1.U(layer1.unpool(layer1.B(layer2.U(layer2.unpool(layer2.B(u22),layer2.index))),layer1.index))#layer1.U(layer1.B(layer2.U(layer2.B(u22))))
            #recon_31  = layer1.U(layer1.unpool(layer1.B(layer2.U(layer2.unpool(layer2.B(layer3.U(x32)),layer2.index))),layer1.index))#layer1.U(layer1.B(layer2.U(layer2.B(layer3.U(x32)))))
            recon_32  = layer1.U(layer1.unpool(layer1.B(layer2.U(layer2.unpool(layer2.B(layer3.U(layer3.unpool(layer3.B(u32),layer3.index))),layer2.index))),layer1.index))
        
        

    # save the results
    for i, image in enumerate(recon_22):
        original_file_path, _ = tiny_imgnet.samples[batch_idx*batch_size+i]
        image = normalize_to_uint8(image)
        u2_img_pil = tensor_to_pil(image.detach().cpu())
        # Get relative path to keep original folder structure
        relative_path = Path(original_file_path).relative_to(data_dir)
        # Create the target folder structure
        target_folder = Path(save_dir_u2) / relative_path.parent
        target_folder.mkdir(parents=True, exist_ok=True)
        
        u2_img_pil.save(target_folder / f"{relative_path.stem}.JPEG")
        
    for i, image in enumerate(recon_32):
        original_file_path, _ = tiny_imgnet.samples[batch_idx*batch_size+i]
        image = normalize_to_uint8(image)
        u3_img_pil = tensor_to_pil(image.detach().cpu())
        # Get relative path to keep original folder structure
        relative_path = Path(original_file_path).relative_to(data_dir)
        # Create the target folder structure
        target_folder = Path(save_dir_u3) / relative_path.parent
        target_folder.mkdir(parents=True, exist_ok=True)
        
        u3_img_pil.save(target_folder / f"{relative_path.stem}.JPEG")