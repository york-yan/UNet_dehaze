import os
from torch import nn,optim
from torch.utils.data import DataLoader

from data import *
from model import unet
import torch
import sys
import torchvision

sys.path.append("../")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
weight_path='./weight/unet.pth'
data_path='./dataset/'
save_path='./train_image/'

if __name__=='__main__':
    data_loader=DataLoader(MyDataset(data_path,mode='train'),batch_size=2,shuffle=True)
    model=unet.UNet().to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('load success')
    else:
        print('no weight')
    
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    loss_func=nn.MSELoss()
    epoch=1
    for epoch in range(10):
        for i,(hazy,clear) in enumerate(data_loader):
            # print(i)
            # print(hazy.shape)
            # print(clear.shape)

            hazy=hazy.to(device)
            clear=clear.to(device)

            out=model(hazy)
            loss=loss_func(out,clear)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%50==0:
                print("i:",i)
                print('epoch:',epoch,'loss:',loss.item())
                torch.save(model.state_dict(),weight_path)
            _hazy=hazy[0]
            _clear=clear[0]
            _out=out[0]
            img=torch.stack([_hazy,_clear,_out],dim=0)
            
            torchvision.utils.save_image(img,f'{save_path}/{i}.png')
        epoch+=1

