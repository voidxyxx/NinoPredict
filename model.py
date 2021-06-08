from torchvision import models
import torch
import torch.nn as nn
import copy


def model_generate(months, outmonths):
    model_resnet = models.resnet34(pretrained=False)
    model_resnet.conv1 = nn.Conv2d(months, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    class unimodel(nn.Module):
        def __init__(self):
            super(unimodel, self).__init__()
            self.modle0 = nn.Sequential(*list(model_resnet.children())[0:-1])
            self.modle1 = nn.Sequential(*copy.deepcopy(list(model_resnet.children())[0:-1]))
            self.modle2 = nn.Sequential(*copy.deepcopy(list(model_resnet.children())[0:-1]))
            self.modle3 = nn.Sequential(*copy.deepcopy(list(model_resnet.children())[0:-1]))
            self.regress = nn.Sequential(nn.Linear(512 * 4, out_features=1000, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(1000, outmonths))

        def forward(self, sst, t300, ua, va):
            sst = self.modle0(sst)
            sst = torch.flatten(sst, 1)
            t300 = self.modle1(t300)
            t300 = torch.flatten(t300, 1)
            ua = self.modle2(ua)
            ua = torch.flatten(ua, 1)
            va = self.modle3(va)
            va = torch.flatten(va, 1)
            combine = torch.cat((sst, t300, ua, va), 1)
            x = self.regress(combine)
            return x

    return unimodel()
