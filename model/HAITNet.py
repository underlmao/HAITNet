import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvt import pvt_v2_b2
from torch.nn.parameter import Parameter
device = torch.device('cuda:0')



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride, groups = groups,
                              padding=padding, dilation=dilation, bias=False)
        self.bn_act = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.bn_act = bn_act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_act(x)
        return x



class FCFM(nn.Module):
    def __init__(self, channel):
        super(FCFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1



class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class GLFEM(nn.Module):
    def __init__(self, num_in=32, mids=4, normalize=False):
        super(GLFEM, self).__init__()

        self.normalize = normalize
        self.num_s = int((mids) * (mids))
        self.num_n = (mids) * (mids)

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # projecttion map
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # learn the reasoning via GCN
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # extend dimention
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)


    def forward(self, x):

        n, c, h, w = x.size()

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped


        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)
        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.conv_extend(x_state)

        return out


        
class GFAM(nn.Module):
    def __init__(self, num_in=32, mids=4, normalize=False):
        super(GFAM, self).__init__()

        self.normalize = normalize
        self.num_s = int((mids) * (mids))
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # projecttion map
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        # learn the reasoning via GCN
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # extend dimention
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):

        n, c, h, w = x.size()
        edge = F.interpolate(edge, x.size()[2:], mode='bilinear')

        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        
        
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))

        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.conv_extend(x_state)

        return out


class HAINet(nn.Module):
    def __init__(self, channel=64):
        super(HAINet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain/pvt_v2_b2.pth'

        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)


        self.FCFM = FCFM(channel)
        self.GFAM = GFAM(channel)
        self.GLFEM = GLFEM(channel)
        self.out_FCFM = nn.Conv2d(channel, 1, 1)
        self.out_GFAM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]


        # Reduce channel
        x1_t = self.Translayer2_0(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        # FCFM
        fcfm_feature = self.FCFM(x4_t, x3_t, x2_t)
        
        #GLFEM
        GLFEM_feature = self.GLFEM(x1_t)
        #GFAM
        GFAM_feature = self.GFAM(fcfm_feature, GLFEM_feature)

        pred_fcfm = self.out_FCFM(fcfm_feature)
        pred_GFAM = self.out_GFAM(GFAM_feature)

        pred_fcfm = F.interpolate(pred_fcfm, size=x.size()[2:], mode='bilinear')
        pred_GFAM = F.interpolate(pred_GFAM, size=x.size()[2:], mode='bilinear')

        if self.training:
            return pred_fcfm, pred_GFAM

        return torch.sigmoid(pred_GFAM)