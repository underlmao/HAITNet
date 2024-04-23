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
import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.backends import cudnn
from model.HAITNet import HAITNet
from dataset import *
from loss import *
from utils import check_mkdir, AvgMeter

cudnn.benchmark = True
torch.manual_seed(2021)
device = torch.device('cuda:0')
ckpt_path = './ckpt'

exp_name = 'PVT_dice'
args = {
    'epoch_num': 100,
    'train_batch_size': 32,
    'last_epoch': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 384,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'AdamW',
}

# config
datasets_root = './data/'

cod_training_root = os.path.join(datasets_root, 'train/') 

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = Compose([RandomHorizontallyFlip(), Resize((args['scale'], args['scale']))
                           ])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness= 0.1, contrast= 0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# data pre-processing
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = structure_loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)
iou_loss = IOU().to(device)
dice_loss = DiceLoss().to(device)

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    out = bce_out + iou_out

    return out


def main():
    print(args)
    print(exp_name)

    net = HAITNet().to(device)
    net.train()
    if args['optimizer'] == 'AdamW':
        print("AdamW")
        optimizer = optim.AdamW([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net)

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record = AvgMeter(), AvgMeter(), AvgMeter()


        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            pred1, pred2 = net(inputs)

            loss_1 = structure_loss(pred1, labels)
            loss_2 = dice_loss(pred2, labels)

            loss = loss_1 + loss_2 * 1.0
            loss.backward()
            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg)

            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.to(device)

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Done!")
            return


if __name__ == '__main__':
    main()
