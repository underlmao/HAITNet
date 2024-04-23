"""
 @Time    : 2021/12/12 15:45
 @Author  : Thanh Hai Phung
 @E-mail  : haipt.eed08g@nctu.edu.tw

 @Project : Mul_COD
 @File    : coarse_net.py
 @Function: ResNet Model File

"""
import datetime
import time
import numpy as np
import os
import torch
from torch import nn
import argparse
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
from utils import AvgMeter, check_mkdir
from config import *
from PIL import ImageFile
from dataset import *
from loss import *
from model.HAITNet import HAITNet
import imageio

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(2021)
# device_ids = 0
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### Main Function #####
parser = argparse.ArgumentParser(description='Decide Which Task to Inference')
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--scale', type=int, default=384)
parser.add_argument('--method', type=str, default='PVT_psu100')
args = parser.parse_args()

save = args.save
save_dir = args.save_dir
scale = args.scale
method = args.method

results_path = args.save_dir
check_mkdir(results_path)


# print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gt_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([('CHAMELEON', chameleon_path),
                       ('CAMO', camo_path),
                       ('COD10K', cod10k_path),
                       ('NC4K', nc4k_path),
                       ])

results = OrderedDict()


def main():
    net = HAITNet().to(device)

    net.load_state_dict(torch.load('./ckpt/' + method + '/100.pth'))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'Imgs')
            if save:
                check_mkdir(os.path.join(results_path, method, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]

            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                w, h = img.size
                img_var = img_transform(img).unsqueeze(0).to(device)
                start_each = time.time()

                pred1, pred2 = net(img_var)
                time_each = time.time() - start_each
                time_list.append(time_each)
                prediction = pred2 
                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

                if save:
                    Image.fromarray(prediction).convert('L').save(
                        os.path.join(results_path, method, name, img_name + '.png'))
            print(('{}'.format(method)))
            print("{}'s average Time Is : {:.3f} s".format(name, np.mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / np.mean(time_list)))
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


if __name__ == '__main__':
    main()