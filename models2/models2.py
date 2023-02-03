import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext, mobilenet, pspnet
from lib.nn import SynchronizedBatchNorm2d
from .pspnet import PSPNet
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F
from .correlation import Correlation
from .function import similarFunction, weightingFunction

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        #valid = (label >= 0).long()
        valid = (label < 255).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc



class SegmentationModule_key(SegmentationModuleBase):
    def __init__(self, low_enc, high_enc, kH = 9, kW = 9):
        super(SegmentationModule_key, self).__init__()
        self.low_encoder = low_enc
        self.high_encoder = high_enc
        self.kH = kH
        self.kW = kW


    def forward(self, feed_dict):
        nonkey_lowfea = self.low_encoder(feed_dict['cur_imgdata']).detach()
        key_lowfea = self.low_encoder(feed_dict['pre_imgdata']).detach()
        key_highfea = self.high_encoder(key_lowfea).detach()
        #print('h, w', key_lowfea.size(2), key_lowfea.size(3), key_highfea.size(2), key_highfea.size(3))
        #batchsize, C, height, width = key_highfea.size()
        #pad = (self.kH // 2, self.kW // 2)
        #key_highfea = F.unfold(key_highfea, kernel_size=(self.kH, self.kW), stride=1, padding=pad).detach()
        #key_highfea = key_highfea.permute(0, 2, 1)
        #key_highfea = key_highfea.permute(0, 2, 1).contiguous()
        #key_highfea = key_highfea.view(batchsize * height * width, C, self.kH * self.kW).detach()

        key_lowfea = key_lowfea.data.cpu()
        key_highfea = key_highfea.data.cpu()
        nonkey_lowfea = nonkey_lowfea.data.cpu()

        return (key_lowfea, key_highfea, nonkey_lowfea)




class SegmentationModule(SegmentationModuleBase):
    def __init__(self, low_enc, high_enc, nonkey_mat, key_sel, key_dec, crit, kH=9, kW=9, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.low_encoder = low_enc
        self.high_encoder = high_enc
        self.key_select = key_sel
        self.key_decoder = key_dec
        self.nonkey_match = nonkey_mat
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.kH = kH
        self.kW = kW

    def forward(self, feed_dict, *, segSize=None):


        #output = self.teacher(feed_dict)
        #return output

        # training
        key_lowfea = self.low_encoder(feed_dict['pre_imgdata']).detach()
        key_highfea = self.high_encoder(key_lowfea).detach()
        key_output = self.key_decoder(key_highfea).detach()

        nonkey_lowfea = self.low_encoder(feed_dict['cur_imgdata']).detach()
        nonkey_highfea = self.high_encoder(nonkey_lowfea).detach()
        nonkey_output = self.key_decoder(nonkey_highfea).detach()

        local_atten = self.nonkey_match(key_lowfea, nonkey_lowfea).detach()
        #print(local_atten.size())
        key_pred = self.key_select(key_lowfea, nonkey_lowfea, local_atten)

        batchsize, C, height, width = key_output.size()
        f_weighting = weightingFunction.apply
        keytrans_output = f_weighting(key_output, local_atten, self.kH, self.kW)


        #pad = (self.kH // 2, self.kW // 2)
        #key_output = F.unfold(key_output, kernel_size=(self.kH, self.kW), stride=1, padding=pad)
        #key_output = key_output.permute(0, 2, 1).contiguous()
        #key_output  = key_output.view(batchsize * height * width, C, self.kH * self.kW)
        #local_atten2 = local_atten.view(batchsize * height * width, self.kH * self.kW, 1)
        #keytrans_output = torch.matmul(key_output, local_atten2)
        #keytrans_output = keytrans_output.squeeze(-1)
        #keytrans_output = keytrans_output.view(batchsize, height, width, C)
        #keytrans_output = keytrans_output.permute(0, 3, 1, 2).contiguous()

        keytrans_output = nn.functional.interpolate(
            keytrans_output, size=(713, 713), mode='bilinear', align_corners=True)
        #keytrans_output = F.log_softmax(keytrans_output, dim=1)
        keytrans_output = F.softmax(keytrans_output, dim=1)
        _, keytrans_preds = torch.max(keytrans_output, dim=1)

        nonkey_output = nn.functional.interpolate(
            nonkey_output, size=(713, 713), mode='bilinear', align_corners=True)
        nonkey_output = F.softmax(nonkey_output, dim=1)
        _, nonkey_preds = torch.max(nonkey_output, dim=1)

        valid = (feed_dict['seg_label'] < 255).long()

        dev_acc_sum = torch.sum(valid* (keytrans_preds != nonkey_preds).long(), dim=(1,2))
        pixel_sum = torch.sum(valid, dim=(1,2))

        dev_acc = dev_acc_sum.float() / (pixel_sum.float() + 1e-10)

        dev_acc = dev_acc.unsqueeze(1)
        #dev_acc = dev_acc.reshape(1).unsqueeze(0)


        loss = self.crit(key_pred, dev_acc)
        #KD_crit = nn.KLDivLoss(reduction='none')
        #KD_deviation = KD_crit(keytrans_output, nonkey_output)
        #KD_deviation = torch.sum(KD_deviation, (1,2,3))
        #KD_deviation = torch.div(KD_deviation, keytrans_output.size(2) * keytrans_output.size(3)).unsqueeze(1)

        #loss = self.crit(key_pred, KD_deviation)
        #print(key_pred.size(), KD_deviation.size(), KD_deviation)

        return (loss)


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Local_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, kH, kW):
        super(Local_Module, self).__init__()
        self.chanel_in = in_dim
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.softmax = Softmax(dim=-1)
        self.kH = kH
        self.kW = kW
        self.m = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)
        #self.f_similar = similarFunction()



    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        m_batchsize, C, height, width = x.size()
        pad = (self.kH // 2, self.kW // 2)

        proj_query = self.query_conv(y)
        proj_key = self.key_conv(x)
        out = self.m(proj_query, proj_key)
        out = out.permute(0, 2, 3, 1)
        out = self.softmax(out)

        #proj_query = self.query_conv(y).permute(0, 2, 3, 1).contiguous()
        #proj_query = proj_query.view(m_batchsize * height * width, 1, C//4)

        #proj_key = self.key_conv(x)
        #proj_key = F.unfold(proj_key, kernel_size=(self.kH, self.kW), stride=1, padding=pad)
        #proj_key = proj_key.contiguous().view(m_batchsize, C//4, self.kH * self.kW, height * width)
        #proj_key = proj_key.permute(0, 3, 1, 2).contiguous()
        #proj_key = proj_key.view(m_batchsize * height * width, C//4, self.kH * self.kW)

        #out = torch.matmul(proj_query, proj_key)
        #out = out.view(m_batchsize, height, width, self.kH * self.kW)
        #out = self.softmax(out)

        return out



class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)


    def build_teacher(self, fc_dim=2048, weights=''):
        #pretrained = True if len(weights) == 0 else False

        net_teacher = PSPNet(layers=50, classes=19, zoom_factor=8, pretrained=False)

        if len(weights) > 0:
            print('Loading weights for net_teacher')
            checkpoint = torch.load(weights)
            net_teacher.load_state_dict(checkpoint['state_dict'], strict=False)
        return net_teacher



    def build_encoder(self, arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'low_resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            encoder = ResnetDilated(orig_resnet, dilate_scale=8)
            encoder.load_state_dict(
                torch.load('./ckpt/image-PSP101/encoder_epoch_40.pth', map_location=lambda storage, loc: storage), strict=False)
            net_encoder = Low_ResnetDilated(encoder)
        elif arch == 'high_resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            encoder = ResnetDilated(orig_resnet, dilate_scale=8)
            encoder.load_state_dict(
                torch.load('./ckpt/image-PSP101/encoder_epoch_40.pth', map_location=lambda storage, loc: storage), strict=False)
            net_encoder = High_ResnetDilated(encoder)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        return net_encoder

    def build_decoder(self, arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'nonkeyc1':
            net_decoder = NonKeyC1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                kH = 9,
                kW = 9)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup_original':
            net_decoder = PPMDeepsupOriginal(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


    def build_match(self, fc_dim=1024, weights='', kH=9, kW=9):
        net_match = LocalAtten(
            fc_dim=fc_dim, kH=kH, kW=kW)
        net_match.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_match')
            net_match.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_match

    def build_keyselect(self, fc_dim=1024, weights='', kH=9, kW=9):
        net_keyselect = KeySelect(
            fc_dim=fc_dim, kH=kH, kW=kW)
        net_keyselect.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_match')
            net_keyselect.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_keyselect



class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class Low_ResnetDilated(nn.Module):
    def __init__(self, encoder):
        super(Low_ResnetDilated, self).__init__()
        from functools import partial

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu1 = encoder.relu1
        self.conv2 = encoder.conv2
        self.bn2 = encoder.bn2
        self.relu2 = encoder.relu2
        self.conv3 = encoder.conv3
        self.bn3 = encoder.bn3
        self.relu3 = encoder.relu3
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = nn.Sequential(*list(encoder.layer3.children())[:4])
        #self.layer4 = orig_resnet.layer4



    def forward(self, x, return_feature_maps=False):
        #conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.layer1(x); conv_out.append(x);
        #x = self.layer2(x); conv_out.append(x);
        #x = self.layer3(x); conv_out.append(x);
        #x = self.layer4(x); conv_out.append(x);
        return x


class High_ResnetDilated(nn.Module):
    def __init__(self, encoder):
        super(High_ResnetDilated, self).__init__()
        from functools import partial
        # take pretrained resnet, except AvgPool and FC

        self.layer3 = nn.Sequential(*list(encoder.layer3.children())[4:])
        self.layer4 = encoder.layer4


    def forward(self, x, return_feature_maps=False):
        #conv_out = []
        #x = self.layer3(x); conv_out.append(x);
        #x = self.layer4(x); conv_out.append(x);

        x = self.layer3(x)
        x = self.layer4(x)

        return x





class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.interpolate(x, size=(713, 713), mode='bilinear', align_corners=True)
        _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


class NonKeyC1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False, kH=9, kW=9):
        super(NonKeyC1, self).__init__()
        self.use_softmax = use_softmax
        self.kH = kH
        self.kW = kW

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 8, 1)
        #self.cbr2 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 8, 1)

        self.lowcbr = conv3x3_bn_relu(fc_dim // 2, fc_dim // 8, 1)
        self.lowcbr2 = conv3x3_bn_relu(fc_dim // 8, fc_dim // 8, 1)
        self.lowcbr3 = conv3x3_bn_relu(fc_dim // 8, fc_dim // 8, 1)


        self.cbr3 = conv3x3_bn_relu(fc_dim// 4, fc_dim //8, 1)
        self.cbr4 = conv3x3_bn_relu(fc_dim // 8, fc_dim // 8, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 8, num_class, 1, 1, 0)
        self.f_weighting = weightingFunction.apply


    def forward(self, low_nonkey, high_key, local_atten, segSize=None):

        #x = high_key[-1]
        #batchsize, height, width, kHW = local_atten.size()
        #x = nn.functional.interpolate(
        #    x, size=(height, width), mode='bilinear', align_corners=False)
        #C = x.size(1)
        high_key = self.cbr(high_key)
        #high_key = self.cbr2(high_key)
        batchsize, C, height, width = high_key.size()
        #print('width, height:', height, width, local_atten.size(1), local_atten.size(2))
        assert high_key.size(2) == local_atten.size(1) and high_key.size(3) == local_atten.size(2)


        #pad = (self.kH // 2, self.kW // 2)
        #x = F.unfold(high_key, kernel_size=(self.kH, self.kW), stride=1, padding=pad)
        #x = x.permute(0, 2, 1).contiguous()
        #x = x.view(batchsize * height * width, C, self.kH * self.kW)
        #local_atten = local_atten.view(batchsize * height * width, self.kH * self.kW, 1)
        #out = torch.matmul(x, local_atten)
        #out = out.squeeze(-1)
        #out = out.view(batchsize, height, width, C)
        #out = out.permute(0, 3, 1, 2).contiguous()

        out = self.f_weighting(high_key, local_atten, self.kH, self.kW)

        y = self.lowcbr(low_nonkey)
        y = self.lowcbr2(y)
        y = self.lowcbr3(y)
        out = torch.cat((out, y), 1)
        out1 = self.cbr3(out)
        out = self.cbr4(out1)


        out = self.conv_last(out)

        out = nn.functional.interpolate(
            out, size=(713, 713), mode='bilinear', align_corners=True)


        if self.use_softmax:  # is True during inference
            out = out
            #out = nn.functional.softmax(out, dim=1)
            return out
        else:
            out = nn.functional.log_softmax(out, dim=1)
            return (out1, out)



class LocalAtten(nn.Module):
    def __init__(self, fc_dim=1024, kH=9, kW=9):
        super(LocalAtten, self).__init__()
        #self.sa = conv3x3_bn_relu(fc_dim, 256, 1)
        #self.sb = conv3x3_bn_relu(fc_dim, 256, 1)
        self.sc = Local_Module(fc_dim, kH, kW)

    def forward(self, key, nonkey):
        x_feat = key
        y_feat = nonkey

        #x_feat = key[-1]
        #y_feat = nonkey[-1]
        #x = self.sa(x_feat)
        #y = self.sb(y_feat)
        z = self.sc(x_feat, y_feat)
        return z


class KeySelect(nn.Module):
    def __init__(self, fc_dim=1024, kH=9, kW=9):
        super(KeySelect, self).__init__()
        self.kH = kH
        self.kW = kW

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr2 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 16, 1)
        self.cbr3 = conv3x3_bn_relu(fc_dim // 16, fc_dim // 4, 1)

        self.cbr4 = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr5 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(fc_dim // 4, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, low_key, low_nonkey, local_atten):

        low_key = self.cbr(low_key)
        low_key = self.cbr2(low_key)

        batchsize, C, height, width = low_key.size()
        assert low_key.size(2) == local_atten.size(1) and low_key.size(3) == local_atten.size(2)
        f_weighting = weightingFunction.apply
        low_transkey = f_weighting(low_key, local_atten, self.kH, self.kW)
        low_transkey = self.cbr3(low_transkey)

        low_nonkey = self.cbr4(low_nonkey)

        out = torch.sub(low_transkey, low_nonkey)
        out = self.cbr5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out



class KeySelect3(nn.Module):
    def __init__(self, fc_dim=1024, kH=9, kW=9):
        super(KeySelect3, self).__init__()
        self.kH = kH
        self.kW = kW

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr2 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 16, 1)
        self.cbr3 = conv3x3_bn_relu(fc_dim // 16, fc_dim // 4, 1)

        self.cbr4 = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr5 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1)

        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(fc_dim // 4, fc_dim // 4)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc_dim // 4, fc_dim // 4)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(fc_dim // 4, 10)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, low_key, low_nonkey, local_atten):

        low_key = self.cbr(low_key)
        low_key = self.cbr2(low_key)

        batchsize, C, height, width = low_key.size()
        assert low_key.size(2) == local_atten.size(1) and low_key.size(3) == local_atten.size(2)
        f_weighting = weightingFunction.apply
        low_transkey = f_weighting(low_key, local_atten, self.kH, self.kW)
        low_transkey = self.cbr3(low_transkey)

        low_nonkey = self.cbr4(low_nonkey)

        out = torch.sub(low_transkey, low_nonkey)
        out = self.cbr5(out)
        out = out.mean(dim=(2,3))
        out = self.fc1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.dropout3(out)
        out = self.fc4(out)

        return out






class KeySelect4(nn.Module):
    def __init__(self, fc_dim=1024, kH=9, kW=9):
        super(KeySelect4, self).__init__()
        self.kH = kH
        self.kW = kW

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr2 = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        self.cbr3 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 4, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(fc_dim // 4, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, low_key, low_nonkey, local_atten):

        low_key = self.cbr(low_key)
        low_nonkey = self.cbr2(low_nonkey)

        batchsize, C, height, width = low_key.size()
        assert low_key.size(2) == local_atten.size(1) and low_key.size(3) == local_atten.size(2)
        f_weighting = weightingFunction.apply
        low_transkey = f_weighting(low_key, local_atten, self.kH, self.kW)


        out = torch.sub(low_transkey, low_nonkey)
        out = self.cbr3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class KeySelect2(nn.Module):
    def __init__(self, fc_dim=1024, kH=9, kW=9):
        super(KeySelect2, self).__init__()
        self.kH = kH
        self.kW = kW

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 8, 1)
        #self.cbr2 = conv3x3_bn_relu(fc_dim // 4, fc_dim // 16, 1)
        #self.cbr3 = conv3x3_bn_relu(fc_dim // 16, fc_dim // 4, 1)

        self.cbr4 = conv3x3_bn_relu(fc_dim, fc_dim // 8, 1)
        self.cbr5 = conv3x3_bn_relu(fc_dim // 8, fc_dim // 8, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(fc_dim // 8, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, low_key, low_nonkey, local_atten):

        low_key = self.cbr(low_key)
        #low_key = self.cbr2(low_key)

        batchsize, C, height, width = low_key.size()
        assert low_key.size(2) == local_atten.size(1) and low_key.size(3) == local_atten.size(2)
        f_weighting = weightingFunction.apply
        low_transkey = f_weighting(low_key, local_atten, self.kH, self.kW)
        #low_transkey = self.cbr3(low_transkey)

        low_nonkey = self.cbr4(low_nonkey)

        out = torch.sub(low_transkey, low_nonkey)
        out = self.cbr5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        #conv5 = conv_out[-1]
        conv5 = conv_out

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            #x = nn.functional.interpolate(
            #    x, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.interpolate(x, size=(713, 713), mode='bilinear', align_corners=True)
        _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


class PPMDeepsupOriginal(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsupOriginal, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        #conv5 = conv_out

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=(713, 713), mode='bilinear', align_corners=True)
            #x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.interpolate(x, size=(713, 713), mode='bilinear', align_corners=True)
        _ = nn.functional.interpolate(_, size=(713, 713), mode='bilinear', align_corners=True)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)



# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
