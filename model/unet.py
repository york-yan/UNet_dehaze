import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Conv_Block(nn.Module):                          # 卷积块，提高网络的表达能力
    def __init__(self,in_channel,out_channel):         # out_channel表示卷积核的数量
        super(Conv_Block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,padding=1),     # 卷积图像输出公式 （H+2*padding-kernel_size)/stride+1
            nn.BatchNorm2d(out_channel),                       # 批归一化，有助于限制梯度大小，避免梯度爆炸
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
    
class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=2,padding=1,padding_mode='reflect',bias=False),  #通道没变，维度减半
            nn.BatchNorm2d(channel),
            nn.LeakyReLU())
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,kernel_size=1,stride=1)      #通道减半
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')                    # 插值
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)     
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.c1=Conv_Block(3,32 //4)
        self.d1=DownSample(32 //4)
        self.c2=Conv_Block(32 //4,64 //4)
        self.d2=DownSample(64 //4)
        self.c3=Conv_Block(64 //4,128 //4)
        self.d3=DownSample(128 //4)
        self.c4=Conv_Block(128 //4,256 //4)
        self.d4=DownSample(256 //4)
        self.c5=Conv_Block(256 //4,512 //4) 
        
        self.u1=UpSample(512 //4)  
        self.c6=Conv_Block(512 //4,256 //4)      
        self.u2=UpSample(256 //4)
        self.c7=Conv_Block(256 //4,128 //4)
        self.u3=UpSample(128 //4)
        self.c8=Conv_Block(128 //4,64 //4)
        self.u4=UpSample(64 //4)
        self.c9=Conv_Block(64 //4,32 //4)
        self.out=nn.Conv2d(32 //4,3,kernel_size=1,stride=1)
        self.Th=nn.Sigmoid()
    
    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1)) 
        R3=self.c3(self.d2(R2))
        R4=self.c4(self.d3(R3))
        R5=self.c5(self.d4(R4))
        print(R5[0,0,0,0])
        O1=self.c6(self.u1(R5,R4))              # self.u1(R5,R4)可以理解为self.u1(R5)+R4 先将R5上采样到R4的大小然后再相加
        O2=self.c7(self.u2(O1,R3))
        O3=self.c8(self.u3(O2,R2))
        O4=self.c9(self.u4(O3,R1))

        out=self.Th(self.out(O4))
        return out


if __name__ == '__main__':

    x=torch.randn(2,3,256,256)

    model=UNet()
    print(model(x).shape)

    # conv_layer=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)

    # input_image=torch.randn(1,3,32,32)
    # print(conv_layer(input_image).shape)
    # output_feature_map=conv_layer(input_image)
    # print(output_feature_map.shape)
    # out=Conv_Block(64,64)
    # print(out(output_feature_map).shape)

    # input_tensor=torch.tensor([-1.0,0.5,1.2])
    # output_tensor1=nn.ReLU()(input_tensor)
    # print(output_tensor1)
    # output_tensor=F.softmax(input_tensor,dim=0)
    # print(output_tensor)