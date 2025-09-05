import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model.Layer1 import Layer1
from model.Layer2 import Layer2
from model.Layer3 import Layer3

from utils.data import ZCA_Loader
import numpy as np
import os

savedir = "runs"
# list subdirectory of "runs", if nothing exists, create a new subdirectory called 0, otherwise, increament the number
subdirs = [int(d) for d in os.listdir(savedir) if os.path.isdir(os.path.join(savedir, d))]
if len(subdirs) == 0:
    subdir = '0'
else:
    subdir = str(max(subdirs) + 1)
savedir = os.path.join(savedir, subdir)
os.makedirs(savedir)
savecause = os.path.join(savedir, 'recon_cause')
os.makedirs(savecause)

zca_data = ZCA_Loader('data/tiny-imagenet-200/zca_images.pt')

batch_size = 32

train_loader = torch.utils.data.DataLoader(
        zca_data, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer1 = Layer1(32, 0, 1, R_lr=0, lmda=0, device=device, save_dir=savedir) # ignore those parameters
layer2 = Layer2(32, 0, 1, R_lr=0, lmda=0, device=device, save_dir=savedir) # ignore those parameters
layer3 = Layer3(32, 0, 1, R_lr=0, lmda=0, device=device, save_dir=savedir) # ignore those parameters

# optimizers
optim_x = torch.optim.SGD([layer1.U.weight], 0.001) #layer1: state dictionary  
optim_a = torch.optim.SGD([layer1.A.weight], 0.001) #layer1: transition dictionary
optim_u = torch.optim.SGD([layer1.B.weight], 0.001) #layer1: cause dictionary
scheduler_c_1 = torch.optim.lr_scheduler.ExponentialLR(optim_x, gamma=0.5)
scheduler_a_1 = torch.optim.lr_scheduler.ExponentialLR(optim_a, gamma=0.5)
scheduler_b_1 = torch.optim.lr_scheduler.ExponentialLR(optim_u, gamma=0.5)

optim_x2 = torch.optim.SGD([layer2.U.weight], 0.001)#layer2: state dictionary
optim_a2 = torch.optim.SGD([layer2.U.weight], 0.001)#layer2: transition dictionary
optim_u2 = torch.optim.SGD([layer2.B.weight], 0.001)#layer2: cause dictionary
scheduler_c_2 = torch.optim.lr_scheduler.ExponentialLR(optim_x2, gamma=0.5)
scheduler_a_2 = torch.optim.lr_scheduler.ExponentialLR(optim_a2, gamma=0.5)
scheduler_b_2 = torch.optim.lr_scheduler.ExponentialLR(optim_u2, gamma=0.5)


optim_x3 = torch.optim.SGD([layer3.U.weight], 0.001)#layer3: state dictionary
optim_a3 = torch.optim.SGD([layer3.A.weight], 0.001)#layer3: transition dictionary
optim_u3 = torch.optim.SGD([layer3.B.weight], 0.001)#layer3: cause dictionary
scheduler_c_3 = torch.optim.lr_scheduler.ExponentialLR(optim_x3, gamma=0.5)
scheduler_a_3 = torch.optim.lr_scheduler.ExponentialLR(optim_a3, gamma=0.5)
scheduler_b_3 = torch.optim.lr_scheduler.ExponentialLR(optim_u3, gamma=0.5)


### record dictionary matrix of states################
C_list = []
C_list2 = []
C_list3 = []

### record loss################
lossx_list = []
lossu_list = []
lossx_list2 = []
lossu_list2 = []
lossx_list3 = []
lossu_list3 = []

Dx = 64
m11 = 768
m12 = 768
m21 = 768
m22 = 768
m31 = 768
m32 = 768

updateiter = 1
for epoch in range(10):
    print('epoch:',epoch)
    for idx, img_batch in enumerate(train_loader):
        img_batch = img_batch.to(device)
        running_loss = 0
        running_lossu = 0
        c = 0
        # repeat same mini-batch
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
                x1, u1, xp1 = layer1.Fista_(img_batch, Rt_1_1, Td_1 , t, )
                img4layer2 = layer1.u.data.clone().detach()
                
                img4layer2_old = layer1.u.data.clone().detach()
                np.save(os.path.join(savedir,'u2.npy'), img4layer2.cpu().detach().numpy())  

                x2, u2, xp2 = layer2.Fista_(img4layer2, Rt_1_2, Td_2, t, )
                img4layer3 = layer2.u.data.clone().detach()
                
                img4layer3_old = layer2.u.data.clone().detach()
                x3, u3, xp3 = layer3.Fista_(img4layer3, Rt_1_3, Td_3, t, )
                np.save(os.path.join(savedir,'u3.npy'), img4layer3.cpu().detach().numpy())
            else:
                x12, u12, xp12 = layer1.Fista_(img_batch, Rt_1_1, Td_1 , t, )
                img4layer2 = layer1.u.data.clone().detach()
                np.save(os.path.join(savedir,'u2.npy'), img4layer2.cpu().detach().numpy())  

                x22, u22, xp22 = layer2.Fista_(img4layer2, Rt_1_2, Td_2, t, )
                img4layer3 = layer2.u.data.clone().detach()
                x32, u32, xp32 = layer3.Fista_(img4layer3, Rt_1_3, Td_3, t, )
                np.save(os.path.join(savedir,'u3.npy'), img4layer3.cpu().detach().numpy())
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
                
                x1 = torch.cat((x1,x12),0)
                u1 = torch.cat((u1,u12),0)
                xp1 = torch.cat((xp1,xp12),0)
                
                x2 = torch.cat((x2,x22),0)
                u2 = torch.cat((u2,u22),0)
                xp2 = torch.cat((xp2,xp22),0)
                
                x3 = torch.cat((x3,x32),0)
                u3 = torch.cat((u3,u32),0)
                xp3 = torch.cat((xp3,xp32),0)
                
                Rt_1_1 = torch.cat((torch.zeros((batch_size, m11, Dx, Dx), device=device),Rt_1_1),0)
                Rt_1_2 = torch.cat((torch.zeros((batch_size, m21, Dx//2, Dx//2), device=device),Rt_1_2),0)
                Rt_1_3 = torch.cat((torch.zeros((batch_size, m31, Dx//4, Dx//4), device=device),Rt_1_3),0)
                
                img_batch = torch.cat((img_batch,img_batch),0)
                img4layer2 = torch.cat((img4layer2_old,img4layer2),0)
                img4layer3 = torch.cat((img4layer3_old,img4layer3),0)
                
                recon_11 = layer1.U(x12)
                recon_12 = layer1.U(layer1.unpool(layer1.B(u12),layer1.index))
                recon_21 = layer1.U(layer1.unpool(layer1.B(layer2.U(x22)),layer1.index))#layer1.U(layer1.B(layer2.U(x22)))
                recon_22 = layer1.U(layer1.unpool(layer1.B(layer2.U(layer2.unpool(layer2.B(u22),layer2.index))),layer1.index))#layer1.U(layer1.B(layer2.U(layer2.B(u22))))
                recon_31  = layer1.U(layer1.unpool(layer1.B(layer2.U(layer2.unpool(layer2.B(layer3.U(x32)),layer2.index))),layer1.index))#layer1.U(layer1.B(layer2.U(layer2.B(layer3.U(x32)))))
                recon_32  = layer1.U(layer1.unpool(layer1.B(layer2.U(layer2.unpool(layer2.B(layer3.U(layer3.unpool(layer3.B(u32),layer3.index))),layer2.index))),layer1.index))
                
                if idx%10 == 0:
            
                    ###record dictionarys###########
                    C_list.append(layer1.U.weight.data.cpu().detach().numpy())
                    np.save(os.path.join(savedir,'C'),C_list[-1])
                    np.save(os.path.join(savedir,'B'),layer1.B.weight.data.cpu().detach().numpy())
                    
                    C_list2.append(layer2.U.weight.data.cpu().detach().numpy())
                    np.save(os.path.join(savedir,'C2'),C_list2[-1])
                    np.save(os.path.join(savedir,'B2'),layer2.B.weight.data.cpu().detach().numpy())
                    
                    C_list3.append(layer3.U.weight.data.cpu().detach().numpy())
                    np.save(os.path.join(savedir,'C3'),C_list3[-1])
                    np.save(os.path.join(savedir,'B3'),layer3.B.weight.data.cpu().detach().numpy())
                    

                    
                    torch.save(recon_11.cpu(), savecause+'/recon11_u.pt')
                    torch.save(recon_12.cpu(), savecause+'/recon12_u.pt')
                    torch.save(recon_21.cpu(), savecause+'/recon21_u.pt')
                    torch.save(recon_22.cpu(), savecause+'/recon22_u.pt')
                    torch.save(recon_31.cpu(), savecause+'/recon31_u.pt')
                    torch.save(recon_32.cpu(), savecause+'/recon32_u.pt')
                    
            u2_show = img4layer2.cpu().detach().clone()
            u2_show[torch.abs(u2_show)>0] = 1
            
            u3_show = img4layer3.cpu().detach().clone()
            u3_show[torch.abs(u3_show)>0] = 1
            
            x2_show = layer1.R.data.cpu().detach().clone()
            x2_show[torch.abs(x2_show)>0] = 1
            
            x3_show = layer2.R.data.cpu().detach().clone()
            x3_show[torch.abs(x3_show)>0] = 1
            
            u4_show = layer3.u.data.cpu().detach().clone()
            u4_show[torch.abs(u4_show)>0] = 1
            
            x4_show = layer3.R.data.cpu().detach().clone()
            x4_show[torch.abs(x4_show)>0] = 1
            
            
            print('sparse:')
            print([u2_show.sum()/len(u2_show.flatten()),
                x2_show.sum()/len(x2_show.flatten()),
                u3_show.sum()/len(u3_show.flatten()),
                x3_show.sum()/len(x3_show.flatten()),
                u4_show.sum()/len(u4_show.flatten()),
                x4_show.sum()/len(x4_show.flatten())])#sparsity: [u1,x1,u2,x2,u3,x3]
            
        for t in range(50):
            #Update dictionary of state in layer1
            for i in range(updateiter): 
                pred, pred_u, pred_xt_1, _ = layer1(img_batch,x1, u1, xp1, Rt_1_1)
                loss = ((img_batch - pred) ** 2).sum()
                running_loss += loss.item()
                loss.backward()
                # update U
                optim_x.step()
                layer1.zero_grad_U()
                layer1.normalize_weights_U() #normalize filters
                if i == 0:
                    lossx_list.append(loss.item())
            
            #Update dictionary of transition in layer1  
            for i in range(updateiter): 
                pred, pred_u, pred_xt_1, _ = layer1(img_batch,x1, u1, xp1, Rt_1_1)
                lossa = pred_xt_1.sum()
                lossa.backward()
                # update A
                optim_a.step()
                layer1.zero_grad_A()
            
            
            #Update dictionary of cause in layer1  
            for i in range(updateiter*5):
                pred, pred_u, pred_xt_1,lossu_show = layer1(img_batch,x1, u1, xp1, Rt_1_1)
                lossu = pred_u.sum()
                running_lossu += lossu.item()
                lossu.backward()
                # update B
                optim_u.step()
                # zero grad
                layer1.zero_grad_B()
                if i == 0:
                    lossu_list.append(lossu_show.sum().item()) 
                # norm
                layer1.normalize_weights_B() #normalize filters
        
            #Update dictionary of state in layer2  
            for i in range(updateiter): 
            
                pred, pred_u,pred_xt_1,_ = layer2(0,x2, u2, xp2, Rt_1_2)
                loss = ((img4layer2 - pred) ** 2).sum()
                running_loss += loss.item()
                loss.backward()
                # update U
                optim_x2.step()
                layer2.zero_grad_U()
                layer2.normalize_weights_U() #normalize filters
                if i == 0:
                    lossx_list2.append(loss.item()) 
            
            #Update dictionary of transition in layer2  
            for i in range(updateiter): 
                pred, pred_u, pred_xt_1,_ = layer2(0,x2, u2, xp2, Rt_1_2)
                lossa = pred_xt_1.sum()
                lossa.backward()
                # update U
                optim_a2.step()
                layer2.zero_grad_A()
            
            #Update dictionary of cause in layer2  
            for i in range(updateiter*10):
                pred, pred_u,pred_xt_1,lossu_show = layer2(0,x2, u2, xp2, Rt_1_2)
                lossu = pred_u.sum()
                running_lossu += lossu.item()
                lossu.backward()
                # update U
                optim_u2.step()
                # zero grad
                layer2.zero_grad_B()
                if i == 0:
                    lossu_list2.append(lossu_show.sum().item()) 
                layer2.normalize_weights_B()#normalize filters
        
            
            #Update dictionary of state in layer3   
            for i in range(updateiter): 
                pred, pred_u,pred_xt_1,_ = layer3(0,x3, u3, xp3, Rt_1_3)
                loss = ((img4layer3 - pred) ** 2).sum()
                running_loss += loss.item()
                loss.backward()
                # update U
                optim_x3.step()
                layer3.zero_grad_U()
                layer3.normalize_weights_U()#normalize filters
                if i == 0:
                    lossx_list3.append(loss.item())
            
            #Update dictionary of transition in layer3  
            for i in range(updateiter):  
                pred, pred_u, pred_xt_1,_ = layer3(0,x3, u3, xp3, Rt_1_3)
                lossa = pred_xt_1.sum()
                lossa.backward()
                # update U
                optim_a3.step()
                layer3.zero_grad_A()
            
            #Update dictionary of cause in layer3  
            for i in range(updateiter*5):
                pred, pred_u,pred_xt_1,lossu_show = layer3(0,x3, u3, xp3, Rt_1_3)
                lossu = pred_u.sum()
                running_lossu += lossu.item()
                lossu.backward()
                # update U
                optim_u3.step()
                # zero grad
                layer3.zero_grad_B()
                if i == 0:
                    lossu_list3.append(lossu_show.sum().item()) 
                layer3.normalize_weights_B()#normalize filters
        
            if t ==0:
                print([epoch, i, lossx_list[-1],lossu_list[-1],lossx_list2[-1],lossu_list2[-1],lossx_list3[-1],lossu_list3[-1]])  
        if idx%10 == 0:
            ###record loss###########
            np.save(os.path.join(savedir,'lossx.npy'),lossx_list)
            np.save(os.path.join(savedir,'lossu.npy'),lossu_list)
            np.save(os.path.join(savedir,'lossx2.npy'),lossx_list2)
            np.save(os.path.join(savedir,'lossu2.npy'),lossu_list2) 
            np.save(os.path.join(savedir,'lossx3.npy'),lossx_list3)
            np.save(os.path.join(savedir,'lossu3.npy'),lossu_list3)
            torch.save(layer1.state_dict(), os.path.join(savedir,'layer1_'+str(1)+'_f9_01.pt'))
            torch.save(layer2.state_dict(), os.path.join(savedir,'layer1_'+str(2)+'_f9_01.pt'))
            torch.save(layer3.state_dict(), os.path.join(savedir,'layer1_'+str(3)+'_f9_01.pt'))