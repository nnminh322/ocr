from torch import nn
from natmin.model.backbone import efficientnet
from natmin.model.backbone import vgg

class CNN(nn.Module):
    def __init__(self,backbone,**kwargs):
        super(CNN,self).__init__()

        if backbone =='efficientnet_b7':
            self.model= efficientnet.efficientnet_b7(**kwargs)
        elif backbone == 'efficient_b0':
            self.model= efficientnet.efficientnet_b0(**kwargs)
        elif backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)


    def forward(self,x):
        return self.model(x)

    def freeze(self,x):
        for name,param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self,x):
        for param in self.model.features.parameters():
            param.requires_grad=True
