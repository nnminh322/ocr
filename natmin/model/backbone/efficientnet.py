import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter

class EfficientNet(nn.Module):
    def __init__(self,name,ss,ks,hidden,pretrained=True,dropout=0.5):
        super(EfficientNet,self).__init__()
        if name == 'efficientnet_b0':
            cnn=models.efficientnet_b7(pretrained=pretrained)
        if name == 'efficientnet_b7':
            cnn=models.efficientnet_b7(pretrained=pretrained)

        pool_idx=0
        for i,layer in enumerate(cnn.features):
            if isinstance(layer,torch.nn.MaxPool2d):
                cnn.features[i]=torch.nn.AvgPool2d(kernel_size=ks[pool_idx],stride=ss[pool_idx],padding=0)
                pool_idx+=1

        self.features=cnn.features
        self.dropout=nn.Dropout(dropout)
        self.last_conv_1x1=nn.Conv2d(512,hidden,1)

    def forward(self,x):
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        conv = conv.transpose(-1,-2)
        conv = conv.flatten(2)
        conv = conv.permute(-1,0,1)

        return conv





def efficientnet_b0(ss,ks,hidden,pretrain=True,dropout=0.5):
    return EfficientNet('efficientnet_b0',ss,ks,hidden,pretrain, dropout)
def efficientnet_b7(ss,ks,hidden,pretrain=True,dropout=0.5):
    return EfficientNet('efficientnet_b7',ss,ks,hidden,pretrain, dropout)
