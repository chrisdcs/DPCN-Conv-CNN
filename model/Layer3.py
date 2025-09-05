import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


m1 = 768
m2 = 768
ksize = 3
kstride = 1

inputsize = 768

batch_size = 32
Dx = 16
class Layer3(nn.Module):

    def __init__(self, N:int, K:int, S:int, R_lr:float=0.1, lmda:float=5e-3, device=None, save_dir=None):
        super(Layer3, self).__init__()
        self.N = N
        self.K = K
        self.S = S
        self.R_lr = R_lr
        self.lmda = lmda
        self.save_dir = save_dir
        # synaptic weights
        self.device = torch.device("cpu") if device is None else device
        self.U = nn.ConvTranspose2d(m1, inputsize, kernel_size=ksize, stride=kstride, padding=ksize//2, bias=False).to(device)
        torch.nn.init.uniform_(self.U.weight,-0.5,0.5)
        self.B = nn.ConvTranspose2d(m2, m1, kernel_size=ksize, stride=kstride, padding=ksize//2, bias=False).to(device)
        torch.nn.init.uniform_(self.B.weight,0,1)
        self.sparse_u = 0.2#0.2#0.35
        self.alphaA = 0.1
        
        self.A = nn.ConvTranspose2d(m1, m1, kernel_size=ksize, stride=kstride, padding=ksize//2, bias=False).to(device)#nn.Linear(Dx*Dx*m1, Dx*Dx*m1,bias=False).to(device)
        torch.nn.init.uniform_(self.U.weight,0.,0.1)
        
        
        
        # responses
        self.R = None
        self.Rz = None
        self.u = torch.zeros((batch_size, m2, int(Dx/2), int(Dx/2)), requires_grad=True, device=self.device)
        self.uz = torch.zeros((batch_size, m2, int(Dx/2), int(Dx/2)), requires_grad=True, device=self.device)
        self.xp = None
        
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        
        # originally 0.1
        self.lamb = 0.2#0.1

        self.normalize_weights_U()
        self.normalize_weights_B()

    def Fista_(self, img_batch, Rt_1, td, t = 0,  td_alpha = 0.01):
        self.alphaA = 0.1
        self.sparse_u = 0.1#0.35
        batch_size = img_batch.shape[0]
        # create R
        #Dx = img_batch.shape[2] 
        #assert (Dx - ksize) % kstride == 0, "Kernel and stride size mismatch"
        #cx = (Dx - ksize) // kstride + 1
        
        #Du = int(cx/2)
        #assert (Du - ksize) % kstride == 0, "Kernel and stride size mismatch"
        #cu = (Du - ksize) // kstride + 1
        
        if t == 0 :
            self.R = torch.zeros((img_batch.shape[0], m1, Dx, Dx), requires_grad=True, device=self.device)
            self.Rz = torch.zeros((img_batch.shape[0], m1, Dx, Dx), requires_grad=True, device=self.device)
            self.u = torch.zeros((img_batch.shape[0], m2, int(Dx/2), int(Dx/2)), requires_grad=True, device=self.device)
            self.uz = torch.zeros((img_batch.shape[0], m2, int(Dx/2), int(Dx/2)), requires_grad=True, device=self.device)
          #td_alpha = 0.0
          #self.alphaA = 0
          
        else:
            self.Rz.data = self.R.data
            self.uz.data = self.u.data
        
        
        old_R = self.R.clone().detach()
        old_Rz = self.Rz.clone().detach()
        old_u = self.u.clone().detach()
        old_uz = self.uz.clone().detach()
        Lx = torch.ones((batch_size, 1 ,1 ,1), device=self.device)
        Lu = torch.ones((batch_size, 1, 1, 1), device=self.device)
        tkx = 1
        tk_1x = 1
        tku = 1
        tk_1u = 1
        mu = 0.01/(m1*Dx*Dx)

       
        # train
        xp = self.R.clone().detach()
        #xp_sign = torch.sign(xp)
        xp,inx = self.maxpool(xp)
        #xp = xp*xp_sign
        self.index = inx

        gama = -self.B(self.u).clone().detach()
        gama = self.unpool(gama,inx)
        gama = ((1+torch.exp(gama))*self.lamb)       
        
        
        lossx = []
        lossu = []
        m4u = 0 
        protect_x = 0
        protect_u = 0
        for m in range(300):
            R_hat = self.A(Rt_1)  
            alpha = (self.Rz - R_hat)/mu
            alpha[alpha>1] = 1
            alpha[alpha<-1] = -1

            
            
            pred_z = self.U(self.Rz)  
            const = ((img_batch - pred_z) ** 2).sum((-1,-2,-3))
            const.sum().backward()
            const = const.data.clone().detach()
            grad_zk = (self.Rz.grad.data.clone().detach()) + (alpha * self.alphaA).data.clone().detach()
            stop_linesearch = torch.zeros((batch_size, 1 ), device=self.device)
            zero_tensor = torch.zeros(self.Rz.shape,device=self.device)
            keep_going = 1
            while keep_going and protect_x<20000:# torch.sum(stop_linesearch) != batch_size:
                gk = self.Rz - grad_zk/Lx
                self.R.data = (torch.sign(gk)*(torch.max((torch.abs(gk)-gama/Lx),zero_tensor))).clone()
                pred = self.U(self.R)
                temp1 = ((img_batch - pred) ** 2).sum((-1,-2,-3)) + (self.alphaA*torch.abs(self.R -  R_hat)).sum((-1,-2,-3))
                temp2 = const + (self.alphaA*torch.abs(self.Rz -  R_hat)).sum((-1,-2,-3)) +((self.R - self.Rz)*grad_zk).sum((-1,-2,-3))+((Lx/2).sum((-1,-2,-3)))*(((self.R - self.Rz)**2).sum((-1,-2,-3)))
                stop_linesearch[temp1<= temp2] = True
                decay = torch.ones((batch_size, 1), device=self.device)
                decay = (1-stop_linesearch)*2
                decay[decay==0] = 1
                Lx = Lx*decay.unsqueeze(-1).unsqueeze(-1)
                protect_x+=1
                if (temp1.sum())<= (temp2.sum()):
                    keep_going = 0
                
            tk_1x = 1+((((m+1))**8-1)/2)
            self.Rz.data = (self.R.clone().detach() + (tkx - 1)/(tk_1x)*(self.R.clone().detach() - old_R)).clone()
            old_R = self.R.clone().detach()
            tkx = tk_1x
            self.zero_grad_x()
            # prox
            lossx.append(temp1.cpu().detach().numpy())
            xp = self.R.clone().detach()
            #xp_sign = torch.sign(xp)
            
            xp,inx = self.maxpool(xp)
            #xp = xp*xp_sign
            self.index = inx
            
            
            for u_iter in range(1):
                pred_uz = (1+torch.exp(-self.B(self.uz)))*self.lamb - self.lamb
                
                const = (pred_uz * torch.abs(xp)).sum((-1,-2,-3)) + (td_alpha*(self.uz - td)**2).sum((-1,-2,-3))
                const.sum().backward()
                const = const.data.clone().detach()
                grad_zk = (self.uz.grad.data.clone().detach())
                stop_linesearch = torch.zeros((batch_size, 1), device=self.device)
                zero_tensor = torch.zeros(self.uz.shape,device=self.device)
                keep_going = 1
                while keep_going and protect_u<20000:#torch.sum(stop_linesearch) != batch_size:
                    gk = self.uz - grad_zk/Lu
                    self.u.data = (torch.sign(gk)*(torch.max((torch.abs(gk)-self.sparse_u/Lu),zero_tensor))).clone()
                    pred = (1+torch.exp(-self.B(self.u)))*self.lamb - self.lamb
                    temp1 = (pred * torch.abs(xp)).sum((-1,-2,-3)) + (td_alpha*(self.u - td)**2).sum((-1,-2,-3))
                    temp2 = const+((self.u - self.uz)*grad_zk).sum((-1,-2,-3))+((Lu/2).sum((-1,-2,-3)))*(((self.u - self.uz)**2).sum((-1,-2,-3)))
                    stop_linesearch[temp1<= temp2] = True
                    decay = torch.ones((batch_size, 1), device=self.device)
                    decay = (1-stop_linesearch)*2
                    decay[decay==0] = 1
                    Lu = Lu*decay.unsqueeze(-1).unsqueeze(-1)
                    protect_u+=1
                  
                    if (temp1.sum())<= (temp2.sum()):
                        keep_going = 0
                self.sparse_u = np.maximum(0.985*self.sparse_u,1e-3)#0.98-9-100
                tk_1u = 1+((((m4u+1))**8-1)/2)
                self.uz.data = (self.u.clone().detach() + (tku - 1)/(tk_1u)*(self.u.clone().detach() - old_u)).clone()
                old_u = self.u.clone().detach()
                tku = tk_1u
        
                lossu.append(temp1.cpu().detach().numpy())
              
                  
                self.zero_grad_u()
                m4u += 1
              
            gama = -self.B(self.u).clone().detach()
            gama = self.unpool(gama,inx)
            gama = ((1+torch.exp(gama))*self.lamb)  
            
        self.xp= xp  
        if not self.save_dir:
            np.save('costx3.npy',lossx)
            np.save('costu3.npy',lossu)    
        else:
            np.save(self.save_dir+'/'+'costx3.npy',lossx)
            np.save(self.save_dir+'/'+'costu3.npy',lossu)
        return self.R.data.detach(), self.u.data.detach(), self.xp.data.detach()

    @staticmethod
    def soft_thresholding_(x, alpha):
        with torch.no_grad():
            rtn = F.relu(x - alpha) - F.relu(-x - alpha)
        return rtn.data

    def zero_grad_x(self):
        self.Rz.grad.zero_()
        
    def zero_grad_u(self):
        self.uz.grad.zero_()
        
    def zero_grad_U(self):
        self.U.zero_grad()
    
    def zero_grad_B(self): 
        self.B.zero_grad()
        
    def zero_grad_A(self): 
        self.A.zero_grad()
    
        
    def normalize_weights_U(self):
        with torch.no_grad():
            self.U.weight.data = self.U.weight.data/(torch.sqrt(torch.sum(self.U.weight.data**2,(1,2,3)))).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            #self.U.weight.data[torch.abs(self.U.weight.data)<0.1**0.5] = 0
    def normalize_weights_B(self):
        with torch.no_grad():
            self.B.weight.data = self.B.weight.data/(torch.sqrt(torch.sum(self.B.weight.data**2,(1,2,3)))).unsqueeze(1).unsqueeze(2).unsqueeze(3) 
            #self.B.weight.data[torch.abs(self.B.weight.data)<0.1**0.5] = 0


    def forward(self, img_batch, x, u, xp, Rt_1):

        pred = self.U(x)
        pred_u = (torch.exp(-self.B(u)))*self.lamb
        pred_xt_1 = torch.abs(x - self.A(Rt_1))
        return pred, pred_u*torch.abs(xp), pred_xt_1,(torch.exp(-self.B(u)))*self.lamb*torch.abs(xp)