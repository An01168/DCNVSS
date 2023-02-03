# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from dataset import TrainDataset, TrainDatasetAgu
from models2 import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
from augmentations import *
import transform
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

# train one epoch
def train(segmentation_module, iterator, optimizer, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()


    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        segmentation_module.zero_grad()

        # forward pass
        (loss) = segmentation_module(batch_data)
        loss = loss.mean()

        # Backward
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_keyselect: {:.6f}, '
                  'Loss: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_keyselect,
                          ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())

        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, args)


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    #(net_encoder, net_decoder, crit) = nets
    (key_select, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)
    dict_keyselect = key_select.state_dict()

    # dict_encoder_save = {k: v for k, v in dict_encoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    # dict_decoder_save = {k: v for k, v in dict_decoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_keyselect,
               '{}/keyselect_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (key_select, crit) = nets
    optimizer_keyselect = torch.optim.SGD(
        group_weight(key_select),
        lr=args.lr_keyselect,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return optimizer_keyselect


def adjust_learning_rate(optimizer, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_keyselect = args.lr_keyselect * scale_running_lr

    #args.running_lr_encoder = args.lr_encoder * scale_running_lr
    #args.running_lr_decoder = args.lr_decoder * scale_running_lr

    optimizer_keyselect = optimizer
    for param_group in optimizer_keyselect.param_groups:
        param_group['lr'] = args.running_lr_keyselect


def main(args):
    # Network Builders
    builder = ModelBuilder()
    low_encoder = builder.build_encoder(
        arch=args.low_encoder,
        fc_dim=args.fc_dim,
        weights='')

    high_encoder = builder.build_encoder(
        arch=args.high_encoder,
        fc_dim=args.fc_dim,
        weights='')

    key_decoder = builder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights='./ckpt/image-PSP101/decoder_epoch_40.pth',
        use_softmax=True)

    nonkey_match = builder.build_match(
        fc_dim=args.fc_dim // 2,
        weights='./ckpt/video-PSP101/match_epoch_40.pth',
        kH = 9,
        kW = 9)

    key_select = builder.build_keyselect(
        fc_dim= args.fc_dim // 2,
        weights= args.weights_keyselect)

    #nonkey_decoder = builder.build_decoder(
    #    arch=args.arch_decoder,
    #    fc_dim=args.fc_dim,
    #    num_class=args.num_class,
    #    weights=args.weights_decoder)

    #net_encoder = builder.build_encoder(
    #    arch=args.arch_encoder,
    #    fc_dim=args.fc_dim,
    #    weights=args.weights_encoder)
    #net_decoder = builder.build_decoder(
    #    arch=args.arch_decoder,
    #    num_class=args.num_class,
    #    weights=args.weights_decoder)

    crit = nn.MSELoss(reduction='mean')


    if args.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            low_encoder, high_encoder, nonkey_match, key_select, key_decoder, crit, 9, 9, args.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            low_encoder, high_encoder, nonkey_match, key_select, key_decoder, crit, 9, 9)

    # Dataset and Loader
    #augment_train = Compose([RandomHorizontallyFlip(), RandomSized((0.5, 0.75)),
    #                         RandomRotate(5), RandomCrop((713, 713))])
    #augment_valid = Compose([RandomHorizontallyFlip(), Scale((args.img_rows, args.img_cols)),
    #                         CenterCrop((448, 896))])

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([0.5, 2.0]),
        transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    dataset_train = TrainDatasetAgu(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu, augmentations=train_transform)
    #dataset_train_org = TrainDataset(
    #    args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)

    #dataset_train = torch.utils.data.ConcatDataset([dataset_train_aug, dataset_train_org])

    #print('# all samples: {}'.format(len(dataset_train)))

    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=len(args.gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(args.gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=args.gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (key_select, crit)
    optimizer = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, iterator_train, optimizer, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args, epoch)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    #parser.add_argument('--arch_encoder', default='resnet101dilated',
    #                    help="architecture of net_encoder")
    parser.add_argument('--low_encoder', default='low_resnet101dilated',
                        help="architecture of low_encoder")
    parser.add_argument('--high_encoder', default='high_resnet101dilated',
                        help="architecture of high_encoder")
    parser.add_argument('--arch_decoder', default='nonkeyc1',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--weights_match', default='',
                        help="weights to finetune net_match")
    parser.add_argument('--weights_keyselect', default='',
                        help="weights to finetune net_keyselect")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/trainCity_video2.odgt')
    parser.add_argument('--list_val',
                        default='./data/validationCity.odgt')
    parser.add_argument('--root_dataset',
                        default='../Cityscapes_video/')
    # optimization related arguments
    parser.add_argument('--gpus', default='0-3',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=3000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    #parser.add_argument('--lr_encoder', default=0.01, type=float, help='LR')
    #parser.add_argument('--lr_decoder', default=0.01, type=float, help='LR')
    parser.add_argument('--lr_match', default=0.01, type=float, help='LR')
    parser.add_argument('--lr_keyselect', default=0.01, type=float, help='LR')
    parser.add_argument('--lr_nonkeydecoder', default=0.01, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=2, type=int,
                        help='number of data loading workers') #16
    parser.add_argument('--imgSize', default=[300, 375, 450, 525, 600],  #[300, 375, 450, 525, 600]
                        nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Parse gpu ids
    all_gpus = parse_devices(args.gpus)
    all_gpus = [x.replace('gpu', '') for x in all_gpus]
    args.gpus = [int(x) for x in all_gpus]
    num_gpus = len(args.gpus)
    args.batch_size = num_gpus * args.batch_size_per_gpu

    args.max_iters = args.epoch_iters * args.num_epoch
    #args.running_lr_encoder = args.lr_encoder
    #args.running_lr_decoder = args.lr_decoder
    args.running_lr_match = args.lr_match
    args.running_lr_keyselect = args.lr_keyselect
    args.running_lr_nonkeydecoder = args.lr_nonkeydecoder

    #args.arch_encoder = args.arch_encoder.lower()
    args.low_encoder = args.low_encoder.lower()
    args.high_encoder = args.high_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()

    # Model ID
    #args.id += '-' + args.arch_encoder
    args.id += '-' + args.low_encoder
    args.id += '-' + args.high_encoder
    args.id += '-' + args.arch_decoder
    args.id += '-ngpus' + str(num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    #args.id += '-LR_encoder' + str(args.lr_encoder)
    #args.id += '-LR_decoder' + str(args.lr_decoder)
    args.id += '-LR_match' + str(args.lr_match)
    args.id += '-LR_keyselect' + str(args.lr_keyselect)
    args.id += '-LR_nonkeydecoder' + str(args.lr_nonkeydecoder)
    args.id += '-epoch' + str(args.num_epoch)
    if args.fix_bn:
        args.id += '-fixBN'
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
