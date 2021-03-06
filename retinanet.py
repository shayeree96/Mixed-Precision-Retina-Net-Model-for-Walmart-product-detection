import torch
import torch.nn as nn

from fpn import FPN50, Res50S16, FPN50O6, FPN101
from torch.autograd import Variable


class TagNet(nn.Module):
    def __init__(self, num_classes=1, num_anchors=1):
        super(TagNet, self).__init__()
        self.fpn = Res50S16()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1,
                                                                      4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1,
                                                                      self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class ProductNet(nn.Module):
    def __init__(self, num_classes=1, num_anchors=1):
        super(ProductNet, self).__init__()
        self.fpn = Res50S16()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1,
                                                                      4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1,
                                                                      self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class RetinaNet(nn.Module):
    def __init__(self, num_classes=20, num_anchors=9, backbone='resnet50'):
        super(RetinaNet, self).__init__()
        if backbone == 'resnet50':
            self.fpn = FPN50()
        elif backbone == 'resnet101':
            self.fpn = FPN101()
        else:
            print('Invalid backbone network')
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class RetinaFaceNet(RetinaNet):
    def __init__(self, num_classes=1, num_anchors=1):
        super(RetinaFaceNet, self).__init__()
        self.fpn = FPN50O6()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)


class RetinaFaceNetVis(RetinaNet):
    def __init__(self, num_classes=1, num_anchors=1):
        super(RetinaFaceNetVis, self).__init__()
        self.fpn = FPN50O6()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return cls_preds


class RetinaFaceNetv2(RetinaNet):
    def __init__(self, num_classes=1, num_anchors=1):
        super(RetinaFaceNetv2, self).__init__()
        self.fpn = FPN50O6()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.loc_head_final = self._make_head(self.num_anchors * 4)
        self.cls_head_final = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for idx, fm in enumerate(fms):
            if idx == len(fms) - 1:
                loc_pred = self.loc_head_final(fm)
                cls_pred = self.cls_head_final(fm)
            else:
                loc_pred = self.loc_head(fm)
                cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        # return cls_preds
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)


def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,224,224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # net = RetinaNet(num_classes=1, num_anchors=1)
    # print net.state_dict().keys()
    net = RetinaFaceNetVis(num_classes=1, num_anchors=1)
    # net.load_state_dict(torch.load('./model/retinaface50.pth'))
    net.load_state_dict(torch.load('./checkpoint/wider/Retina50FLO6/ckpt_best.pth')['net'])
    x = Variable(torch.zeros((1, 3, 1024, 1024)), requires_grad=True)
    print(x.grad)
    cls_preds = net(x)
    for cls in cls_preds:
        print(cls.size())
    l = 5
    pos = 4
    grad = torch.zeros(cls_preds[l].size())
    grad[0, 0, pos, pos] = 1.0
    cls_preds[l].backward(grad)
    rf = x.grad.data
    im = rf.squeeze().permute(1, 2, 0).norm(p=2, dim=2, keepdim=True)
    print(rf[0, 0])
    plt.imshow(im.squeeze().numpy())
    plt.show()
