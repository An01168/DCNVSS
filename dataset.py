import os
import json
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
from matplotlib.pyplot import imread
import numpy as np
from augmentations import *


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

class TrainDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1, batch_per_gpu=1):
        self.root_dataset = opt.root_dataset
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        self.random_flip = opt.random_flip
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[72.39, 73.16, 82.19], std=[255., 255., 255.])
            ])

        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        self.if_shuffled = False

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))


        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width), \
                    self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate,\
                'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_segms = torch.zeros(self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate, \
                                batch_resized_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            img = imread(image_path, mode='RGB')
            segm = imread(segm_path)

            segm = self.encode_segmap(np.array(segm, dtype=np.uint8))

            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

            if self.random_flip == True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)

            # note that each sample within a mini batch has different scale param
            img = cv2.resize(img, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interpolation=cv2.INTER_LINEAR)
            segm = cv2.resize(segm, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interpolation=cv2.INTER_NEAREST)

            # to avoid seg label misalignment
            segm_rounded_height = round2nearest_multiple(segm.shape[0], self.segm_downsampling_rate)
            segm_rounded_width = round2nearest_multiple(segm.shape[1], self.segm_downsampling_rate)
            segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            segm_rounded[:segm.shape[0], :segm.shape[1]] = segm

            segm = cv2.resize(segm_rounded, (segm_rounded.shape[0] // self.segm_downsampling_rate, \
                                             segm_rounded.shape[1] // self.segm_downsampling_rate), \
                              interpolation=cv2.INTER_NEAREST)
             # image to float
            img = img.astype(np.float32)[:, :, ::-1] # RGB to BGR!!!
            img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))

            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = batch_segms  # label from -1 to 149
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


class TrainDatasetAgu(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1, batch_per_gpu=1, augmentations=None):
        self.root_dataset = opt.root_dataset
        #self.imgSize = opt.imgSize
        #self.imgMaxSize = opt.imgMaxSize
        #self.random_flip = opt.random_flip
        # max down sampling rate of network to avoid rounding during conv or pooling
        #self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        self.augmentations = augmentations

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        #self.img_transform = transforms.Compose([
        #    transforms.Normalize(mean=[72.39, 73.16, 82.19], std=[255., 255., 255.])
        #    ])

        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        self.if_shuffled = False

        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        #self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        #self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
        #                    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
        #                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        #                    'motorcycle', 'bicycle']

        self.ignore_index = 255
        #self.class_map = dict(zip(self.valid_classes, range(19)))


        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        #if not self.if_shuffled:
        #    np.random.shuffle(self.list_sample)
        #    self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()


        batch_resized_height = 713
        batch_resized_width = 713
        batch_pre_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_cur_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)

        #batch_segms = torch.zeros(self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate, \
        #                          batch_resized_width // self.segm_downsampling_rate).long()

        batch_segms = torch.zeros(self.batch_per_gpu, batch_resized_height, batch_resized_width).long()


        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            #pre_image_path = os.path.join(self.root_dataset, this_record['pre_img'])
            cur_image_path = os.path.join(self.root_dataset, this_record['cur_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            cur_img_name = this_record['cur_img'].split('_')
            cur_img_num = int(cur_img_name[3])
            pre_img_index = random.randint(1, 10)
            pre_img_num = str(cur_img_num-pre_img_index).zfill(6)
            pre_img_name = cur_img_name[0] + '_'+ cur_img_name[1]+ '_'+ cur_img_name[2]+ '_'+ pre_img_num+ '_leftImg8bit.png'
            pre_image_path = os.path.join(self.root_dataset, pre_img_name)

            pre_img = cv2.imread(pre_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            pre_img = np.float32(pre_img)

            cur_img = cv2.imread(cur_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            cur_img = np.float32(cur_img)

            segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

            #segm = self.encode_segmap(np.array(segm, dtype=np.uint8))

            assert(cur_img.ndim == 3)
            assert(pre_img.ndim == 3)
            assert(segm.ndim == 2)
            assert(pre_img.shape[0] == segm.shape[0])
            assert(pre_img.shape[1] == segm.shape[1])
            assert(cur_img.shape[0] == segm.shape[0])
            assert(cur_img.shape[1] == segm.shape[1])

            imgs = []
            imgs.append(pre_img)
            imgs.append(cur_img)

            if self.augmentations is not None:
                imgs, segm = self.augmentations(imgs, segm)


            batch_pre_images[i][:, :segm.shape[0], :segm.shape[1]] = imgs[0]
            batch_cur_images[i][:, :segm.shape[0], :segm.shape[1]] = imgs[1]

            #batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm
            #batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        #batch_segms = batch_segms  # label from -1 to 149
        output = dict()
        output['pre_imgdata'] = batch_pre_images
        output['cur_imgdata'] = batch_cur_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass




class ValDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1, start_idx=-1, end_idx=-1):
        self.root_dataset = opt.root_dataset
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # mean and std
        #self.img_transform = transforms.Compose([
        #    transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        #    ])

        self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        #self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        #self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
        #                    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
         #                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
         #                   'motorcycle', 'bicycle']

        self.ignore_index = 255
        #self.class_map = dict(zip(self.valid_classes, range(19)))

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        if start_idx >= 0 and end_idx >= 0: # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label

        pre_image_path = os.path.join(self.root_dataset, this_record['pre_img'])
        cur_image_path = os.path.join(self.root_dataset, this_record['cur_img'])
        #image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

        pre_img = cv2.imread(pre_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        cur_img = cv2.imread(cur_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        #segm = imread(segm_path)

        #segm = self.encode_segmap(np.array(segm, dtype=np.uint8))

        #ori_height, ori_width, _ = img.shape


        #img_resized_list = []
        pre_img_resized = pre_img.astype(np.float32)
        pre_img_resized = pre_img_resized.transpose((2, 0, 1))
        pre_img_resized = torch.from_numpy(pre_img_resized)
        if not isinstance(pre_img_resized, torch.FloatTensor):
            pre_img_resized = pre_img_resized.float()
        pre_img_resized = torch.unsqueeze(pre_img_resized, 0)

        cur_img_resized = cur_img.astype(np.float32)
        cur_img_resized = cur_img_resized.transpose((2, 0, 1))
        cur_img_resized = torch.from_numpy(cur_img_resized)
        if not isinstance(cur_img_resized, torch.FloatTensor):
            cur_img_resized = cur_img_resized.float()
        cur_img_resized = torch.unsqueeze(cur_img_resized, 0)


        segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        #print(np.unique(segm.astype(np.int)))


        #img_resized = cv2.resize(img.copy(), (713, 713))
        #img_resized = img_resized.astype(np.float32)
        #img_resized = img_resized.transpose((2, 0, 1))
        #img_resized = self.img_transform(torch.from_numpy(img_resized))
        #img_resized = torch.unsqueeze(img_resized, 0)
        #img_resized_list.append(img_resized)


        #for this_short_size in self.imgSize:
            # calculate target height and width
        #    scale = min(this_short_size / float(min(ori_height, ori_width)),
        #            self.imgMaxSize / float(max(ori_height, ori_width)))
        #    target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
        #    target_height = round2nearest_multiple(target_height, self.padding_constant)
        #    target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
        #    img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
        #    img_resized = img_resized.astype(np.float32)
        #    img_resized = img_resized.transpose((2, 0, 1))
        #    img_resized = self.img_transform(torch.from_numpy(img_resized))

        #    img_resized = torch.unsqueeze(img_resized, 0)
        #    img_resized_list.append(img_resized)

        segm = torch.from_numpy(segm.astype(np.int)).long()


        if not isinstance(segm, torch.LongTensor):
            segm = segm.long()

        batch_segms = torch.unsqueeze(segm, 0)

        #batch_segms = batch_segms   # label from -1 to 149
        output = dict()
        output['img_ori'] = cur_img.copy()
        #output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['pre_imgdata'] = pre_img_resized
        output['cur_imgdata'] = cur_img_resized
        output['seg_label'] = batch_segms
        output['info'] = this_record['pre_img']
        output['cur_info'] = this_record['cur_img']
        return output

    def __len__(self):
        return self.num_sample

    #def encode_segmap(self, mask):
        # Put all void classes to zero
    #    for _voidc in self.void_classes:
    #        mask[mask == _voidc] = self.ignore_index
    #    for _validc in self.valid_classes:
    #        mask[mask == _validc] = self.class_map[_validc]
    #    return mask


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(torchdata.Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label



class TestDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1] # BGR to RGB!!!

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm = torch.from_numpy(segm.astype(np.int)).long()

        # batch_segms = torch.unsqueeze(segm, 0)

        # batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
