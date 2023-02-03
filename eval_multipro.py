# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import ValDataset, SemData
from models import ModelBuilder, SegmentationModule
from models.pspnet import PSPNet
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, colorize
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
import torch.nn.functional as F
import transform
import torch.backends.cudnn as cudnn
import timeit
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

colors = loadmat('data/color19.mat')['col']


def visualize_result(data, pred, args):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)
    #pred_color = colorize(pred, colors)

    # aggregate images and save
    #im_vis = np.concatenate((img, seg_color, pred_color),
    #                        axis=1).astype(np.uint8)

    im_vis = np.concatenate((seg_color, pred_color),
                             axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)
    #color_path = os.path.join(args.result, img_name.replace('.jpg', '.png'))
    #pred_color.save(color_path)




def net_process(low_encoder, high_encoder, nonkey_match, nonkey_decoder, pre_image, cur_image, mean, std=None, flip=True):
    pre_input = torch.from_numpy(pre_image.transpose((2, 0, 1))).float()
    cur_input = torch.from_numpy(cur_image.transpose((2, 0, 1))).float()
    if std is None:
        for t1, m1 in zip(pre_input, mean):
            t1.sub_(m1)
        for t2, m2 in zip(cur_input, mean):
            t2.sub_(m2)
    else:
        for t1, m1, s1 in zip(pre_input, mean, std):
            t1.sub_(m1).div_(s1)
        for t2, m2, s2 in zip(cur_input, mean, std):
            t2.sub_(m2).div_(s2)
    pre_input = pre_input.unsqueeze(0).cuda()
    cur_input = cur_input.unsqueeze(0).cuda()
    if flip:
        pre_input = torch.cat([pre_input, pre_input.flip(3)], 0)
        cur_input = torch.cat([cur_input, cur_input.flip(3)], 0)
    with torch.no_grad():

        key_lowfea = low_encoder(pre_input)
        key_highfea = high_encoder(key_lowfea)
        nonkey_lowfea = low_encoder(cur_input)
        local_atten = nonkey_match(key_lowfea, nonkey_lowfea)

        output = nonkey_decoder(nonkey_lowfea, key_highfea, local_atten)



        #output1 = net_encoder(input)
        #output = net_decoder(output1)
        #output = model(input)
    _, _, h_i, w_i = pre_input.shape
    _, _, h_o, w_o = output.shape

    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]



    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(low_encoder, high_encoder, nonkey_match, nonkey_decoder, pre_image, cur_image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = pre_image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        pre_image = cv2.copyMakeBorder(pre_image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
        cur_image = cv2.copyMakeBorder(cur_image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = pre_image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            pre_image_crop = pre_image[s_h:e_h, s_w:e_w].copy()
            cur_image_crop = cur_image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(low_encoder, high_encoder, nonkey_match, nonkey_decoder, pre_image_crop, cur_image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def net_process2(net_encoder, net_decoder, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output1 = net_encoder(input)
        output = net_decoder(output1)
        #output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape

    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output



def scale_process2(net_encoder, net_decoder, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process2(net_encoder, net_decoder, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def evaluate(low_encoder, high_encoder, nonkey_match, nonkey_decoder, key_encoder, key_decoder, loader, args, result_queue):
    #segmentation_module.eval()
    #net_encoder.eval()
    #net_decoder.eval()
    low_encoder.eval()
    high_encoder.eval()
    nonkey_match.eval()
    nonkey_decoder.eval()
    key_encoder.eval()
    key_decoder.eval()

    crop_h = 713
    crop_w = 713

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    j = 0

    #for i, (input, _) in enumerate(loader):
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])

        j = j + 1


        #input = batch_data['img_data']
        pre_input = batch_data['pre_imgdata']
        pre_input= np.squeeze(pre_input.numpy(), axis=0)
        pre_image = np.transpose(pre_input, (1, 2, 0))

        cur_input = batch_data['cur_imgdata']
        cur_input= np.squeeze(cur_input.numpy(), axis=0)
        cur_image = np.transpose(cur_input, (1, 2, 0))


        #input = np.squeeze(input.numpy(), axis=0)
        #image = np.transpose(input, (1, 2, 0))
        h, w, _ = pre_image.shape
        prediction = np.zeros((h, w, args.num_class), dtype=float)
        cur_img_name = batch_data['cur_info'].split('/')[-1].rstrip()
        pre_img_name = batch_data['info'].split('/')[-1].rstrip()
        if cur_img_name == pre_img_name:
            for scale in args.scales:
                long_size = round(scale * 2048)
                new_h = long_size
                new_w = long_size
                if h > w:
                    new_w = round(long_size / float(h) * w)
                else:
                    new_h = round(long_size / float(w) * h)
                #pre_image_scale = cv2.resize(pre_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                cur_image_scale = cv2.resize(cur_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += scale_process2(key_encoder, key_decoder, cur_image_scale, args.num_class, crop_h, crop_w, h, w, mean, std)
        else:
            for scale in args.scales:
                long_size = round(scale * 2048)
                new_h = long_size
                new_w = long_size
                if h > w:
                    new_w = round(long_size / float(h) * w)
                else:
                    new_h = round(long_size / float(w) * h)
                pre_image_scale = cv2.resize(pre_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                cur_image_scale = cv2.resize(cur_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += scale_process(low_encoder, high_encoder, nonkey_match, nonkey_decoder, pre_image_scale, cur_image_scale, args.num_class, crop_h, crop_w, h, w, mean, std)

        prediction /= len(args.scales)
        #print('j:', j)
        prediction = np.argmax(prediction, axis=2)
        pred = as_numpy(prediction)
        a = np.unique(prediction)
        print('j:', j)

        #with torch.no_grad():
        #    segSize = (seg_label.shape[0], seg_label.shape[1])
        #    scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        #    scores = async_copy_to(scores, gpu_id)

            #for img in img_resized_list:
            #    feed_dict = batch_data.copy()
            #    feed_dict['img_data'] = img
            #    del feed_dict['img_ori']
            #    del feed_dict['info']
            #    feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
            #    scores_tmp = segmentation_module(feed_dict, segSize=segSize)
            #    scores = scores + scores_tmp / len(args.imgSize)

            #_, pred = torch.max(scores, dim=1)
            #pred = as_numpy(pred.squeeze(0).cpu())
            #a = np.unique(pred)
            #print('a:', a)

        #print('a:', a)

        # calculate accuracy and SEND THEM TO MASTER
        #acc, pix = accuracy(prediction, seg_label)
        intersection, union, target = intersectionAndUnion(prediction, seg_label, args.num_class)
        #result_queue.put_nowait((acc, pix, intersection, union))
        result_queue.put_nowait((intersection, union, target))

        # visualization
        if args.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                prediction, args)




def worker(args, result_queue):
    #torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        args.list_val, args, max_sample=args.num_val,
        start_idx=-1, end_idx=-1)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)


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

    key_encoder = builder.build_encoder(
        arch='resnet101dilated',
        fc_dim=args.fc_dim,
        weights='./ckpt/image-PSP101/encoder_epoch_40.pth')

    key_decoder = builder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights='./ckpt/image-PSP101/decoder_epoch_40.pth',
        use_softmax=True)

    nonkey_match = builder.build_match(
        fc_dim=args.fc_dim // 2,
        weights=args.weights_match,
        kH = 9,
        kW = 9)

    nonkey_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)
    

    crit = nn.NLLLoss(ignore_index=255)

    #segmentation_module = net_teacher
    #segmentation_module = torch.nn.DataParallel(segmentation_module).cuda()
    #cudnn.benchmark = True

    #checkpoint = torch.load(args.weights_teacher)
    #segmentation_module.load_state_dict(checkpoint['state_dict'], strict=False)

    #net_encoder = torch.nn.DataParallel(net_encoder).cuda()
    #net_decoder = torch.nn.DataParallel(net_decoder).cuda()
    low_encoder = torch.nn.DataParallel(low_encoder).cuda()
    high_encoder = torch.nn.DataParallel(high_encoder).cuda()
    nonkey_match = torch.nn.DataParallel(nonkey_match).cuda()
    nonkey_decoder = torch.nn.DataParallel(nonkey_decoder).cuda()
    key_encoder = torch.nn.DataParallel(key_encoder).cuda()
    key_decoder = torch.nn.DataParallel(key_decoder).cuda()
    cudnn.benchmark = True


    #net_encoder.load_state_dict(
    #    torch.load(args.weights_encoder, map_location=lambda storage, loc: storage), strict=False)
    #net_decoder.load_state_dict(
    #    torch.load(args.weights_decoder, map_location=lambda storage, loc: storage), strict=False)

    # Main loop
    evaluate(low_encoder, high_encoder, nonkey_match, nonkey_decoder, key_encoder, key_decoder, loader_val, args, result_queue)


def main(args):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    intersection_delta = 0.01

    with open(args.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)
    print('num:',num_files)

    result_queue = Queue(2501)

    worker(args, result_queue)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (intersection, union, target) = result_queue.get()
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        processed_counter += 1


    # summary
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) + intersection_delta
    mIoU = np.mean(iou_class)


    for i in range(args.num_class):
        print('class [{}], IoU: {:.4f}'.format(i, iou_class[i]))


    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}'.format(mIoU))

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")
    parser.add_argument('--low_encoder', default='low_resnet101dilated',
                        help="architecture of low_encoder")
    parser.add_argument('--high_encoder', default='high_resnet101dilated',
                        help="architecture of high_encoder")
    parser.add_argument('--arch_decoder', default='nonkeyc1',
                        help="architecture of net_decoder")
    #parser.add_argument('--arch_encoder', default='resnet50dilated',
    #                    help="architecture of net_encoder")
    #parser.add_argument('--arch_decoder', default='ppm_deepsup',
    #                    help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--scales', default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], type=int)

    # Path related arguments
    parser.add_argument('--list_val',
                        default='./data/valCity_video_select.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/Cityscapes_video/')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[713], nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g.  300 400 500 600')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')

    # Misc arguments
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpus', default='0',
                        help='gpu ids for evaluation')

    args = parser.parse_args()
    args.low_encoder = args.low_encoder.lower()
    args.high_encoder = args.high_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    #args.weights_encoder = os.path.join(args.ckpt, args.id,
    #                                    'encoder' + args.suffix)
    #args.weights_decoder = os.path.join(args.ckpt, args.id,
    #                                    'decoder' + args.suffix)


    args.weights_decoder = os.path.join(args.ckpt, args.id,
                                        'nonkeydecoder' + args.suffix)
    args.weights_match = os.path.join(args.ckpt, args.id, 'match' +  args.suffix)

    #assert os.path.exists(args.weights_decoder) and \
    #    os.path.exists(args.weights_decoder), 'checkpoint does not exitst!'
    assert os.path.exists(args.weights_match), 'checkpoint does not exitst!'
    assert os.path.exists(args.weights_decoder), 'checkpoint does not exitst!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
