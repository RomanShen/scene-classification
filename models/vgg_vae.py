import torch
from .modules import Classifier
from .modules import MyVgg16
from .modules import VAEDecoder
from torch import nn


class VggVAE(nn.Module):
    def __init__(self, num_classes=45):
        super(VggVAE, self).__init__()
        self.vgg = MyVgg16()
        self.classifier = Classifier(num_classes)
        self.vae_decoder = VAEDecoder()

    def forward(self, x):
        z, features = self.vgg(x)
        x_recon, mu, logvar = self.vae_decoder(z)
        features = self.classifier(features)
        return features, x_recon, mu, logvar


def create_VggVAE(pretrained_weights=None, pre_num_classes=45, cur_num_classes=45):
    if pretrained_weights:
        vgg_vae = VggVAE(pre_num_classes)
        vgg_vae.load_state_dict(torch.load(pretrained_weights))
        vgg_vae.classifier = Classifier(cur_num_classes)
    else:
        vgg_vae = VggVAE(cur_num_classes)
    return vgg_vae


if __name__ == '__main__':
    import sys
    # print(sys.modules)
    # vgg = VggVAE(num_classes=45)
    # output = vgg(torch.rand(4, 3, 224, 224))
    # for i in output:
    #     print(i.shape)
    # print(vgg.state_dict().keys())
    vv = create_VggVAE("123")
