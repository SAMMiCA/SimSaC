"""
Extracted from DGC-Net https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
"""
from os import path as osp
import os
import cv2
import flow_vis
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.util import center_crop
from matplotlib import pyplot as plt
import albumentations as A
from utils.copy_paste import CopyPaste
from utils.coco import CocoDetectionCP
import random

def unormalise_and_convert_mapping_to_flow(map, output_channel_first=True):

    if not isinstance(map, np.ndarray):
        #torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            B, C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[1, :, :] = (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        flow = np.copy(map)
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.permute(0, 2, 3, 1)

            #BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            mapping[:,:,:,0] = (map[:,:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,:,1] = (map[:,:,:,1] + 1) * (h_scale - 1) / 2
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:,:,0] = (map[:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,1] = (map[:,:,1] + 1) * (h_scale - 1) / 2
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = mapping[:,:,0]-X
            flow[:,:,1] = mapping[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1).float()
        return flow.astype(np.float32)

def cxcywh2xyxy(cxcywh):
    if len(cxcywh) == 4:
        x, y, w, h = cxcywh
    elif len(cxcywh) == 5:
        x, y, w, h, class_idx = cxcywh
    elif len(cxcywh) == 6:
            x, y, w, h, class_idx, bbox_idx = cxcywh
    else:
        raise ValueError

    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2

    if len(cxcywh) == 4:
        return (x1, y1, x2, y2)
    elif len(cxcywh) == 5:
        return (x1, y1, x2, y2, class_idx)
    elif len(cxcywh) == 6:
        return (x1, y1, x2, y2, class_idx, bbox_idx)


class HomoAffTps_Dataset(Dataset):
    """
    Extracted from https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
    Main dataset for generating the training/validation the proposed approach.
    It can handle affine, TPS, and Homography transformations.
    Args:
        image_path: filepath to the dataset
        csv_file: csv file with ground-truth transformation parameters and name of original images
        transforms: image transformations for the source image (data preprocessing)
        transforms_target: image transformations for the target image (data preprocessing), if different than that of
        the source image
        get_flow: bool, whether to get flow or normalized mapping
        pyramid_param: spatial resolution of the feature maps at each level
            of the feature pyramid (list)
        output_size: size (tuple) of the output images
    Output:
        if get_flow:
            source_image: source image, shape 3xHxWx
            target_image: target image, shape 3xHxWx
            flow_map: corresponding ground-truth flow field, shape 2xHxW
            correspondence_mask: mask of valid flow, shape HxW
        else:
            source_image: source image, shape 3xHxWx
            target_image: target image, shape 3xHxWx
            correspondence_map: correspondence_map, normalized to [-1,1], shape HxWx2,
                                should correspond to correspondence_map_pyro[-1]
            correspondence_map_pyro: pixel correspondence map for each feature pyramid level
            mask_x: X component of the mask (valid/invalid correspondences)
            mask_y: Y component of the mask (valid/invalid correspondences)
            correspondence_mask: mask of valid flow, shape HxW, equal to mask_x and mask_y
    """

    def __init__(self,
                 image_path,
                 csv_file,
                 transforms,
                 transforms_target=None,
                 get_flow=False,
                 pyramid_param=[520],
                 output_size=(520,520)):
        super().__init__()
        self.img_path = image_path
        if not os.path.isdir(self.img_path):
            raise ValueError("The image path that you indicated does not exist!")

        self.transform_dict = {0: 'aff', 1: 'tps', 2: 'homo'}
        self.transforms_source = transforms
        if transforms_target is None:
            self.transforms_target = transforms
        else:
            self.transforms_target = transforms_target
        self.pyramid_param = pyramid_param
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            if len(self.df) == 0:
                raise ValueError("The csv file that you indicated is empty !")
        else:
            raise ValueError("The path to the csv file that you indicated does not exist !")

        self.H_OUT, self.W_OUT = output_size

        # changed compared to version from DGC-Net
        self.ratio_cropping = 1.5
        # this is a scaling to apply to the homographies, usually applied to get 240x240 images
        self.ratio_TPS = self.H_OUT / 240.0
        self.ratio_homography = self.H_OUT / 240.0

        self.H_AFF_TPS, self.W_AFF_TPS = (int(480*self.ratio_TPS), int(640*self.ratio_TPS))
        self.H_HOMO, self.W_HOMO = (int(576*self.ratio_homography), int(768*self.ratio_homography))

        self.THETA_IDENTITY = \
            torch.Tensor(np.expand_dims(np.array([[1, 0, 0],
                                                  [0, 1, 0]]),
                                        0).astype(np.float32))
        self.gridGen = TpsGridGen(self.H_OUT, self.W_OUT)

        self.copy_paste_transform_prob50 = A.Compose([
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)
        ], bbox_params = A.BboxParams(format="coco",min_visibility=0.3))
        self.copy_paste_transform_prob100 = A.Compose([
            CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1)
        ], bbox_params = A.BboxParams(format="coco",min_visibility=0.3))
        self.coco_data = CocoDetectionCP(
            '../dataset/coco/train2014/',
            '../dataset/coco/annotations/instances_train2014.json'
        )

        self.rescale_transform = A.Compose([
            A.LongestMaxSize(max_size=200)
        ])
        self.rotate_transform = A.Compose([
            A.SafeRotate(limit=360,mask_value=0,p=0.7,border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.6),
            A.Transpose(p=0.6)
        ])

    def transform_image(self,
                        image,
                        out_h,
                        out_w,
                        padding_factor=1.0,
                        crop_factor=1.0,
                        theta=None):
        sampling_grid = self.generate_grid(out_h, out_w, theta)
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        if float(torch.__version__[:3]) >= 1.3:
            warped_image_batch = F.grid_sample(image, sampling_grid, align_corners=True)
        else:
            warped_image_batch = F.grid_sample(image, sampling_grid)
        return warped_image_batch

    def generate_grid(self, out_h, out_w, theta=None):
        out_size = torch.Size((1, 3, out_h, out_w))
        if theta is None:
            theta = self.THETA_IDENTITY
            theta = theta.expand(1, 2, 3).contiguous()
            return F.affine_grid(theta, out_size)
        elif (theta.shape[1] == 2):
            return F.affine_grid(theta, out_size)
        else:
            return self.gridGen(theta)

    def get_grid(self, H, ccrop):
        # top-left corner of the central crop
        X_CCROP, Y_CCROP = ccrop[0], ccrop[1]

        W_FULL, H_FULL = (self.W_HOMO, self.H_HOMO)
        W_SCALE, H_SCALE = (self.W_OUT, self.H_OUT)

        # inverse homography matrix
        Hinv = np.linalg.inv(H)
        Hscale = np.eye(3)
        Hscale[0,0] = Hscale[1,1] = self.ratio_homography
        Hinv = Hscale @ Hinv @ np.linalg.inv(Hscale)

        # estimate the grid for the whole image
        X, Y = np.meshgrid(np.linspace(0, W_FULL - 1, W_FULL),
                           np.linspace(0, H_FULL - 1, H_FULL))
        X_, Y_ = X, Y
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        X_grid_pivot = (XwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)
        Y_grid_pivot = (YwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)

        # normalize XwarpHom and YwarpHom and cast to [-1, 1] range
        Xwarp = (2 * X_grid_pivot / (W_FULL - 1) - 1)
        Ywarp = (2 * Y_grid_pivot / (H_FULL - 1) - 1)
        grid_full = torch.stack([Xwarp, Ywarp], dim=-1)

        # getting the central patch from the pivot
        Xwarp_crop = X_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        Ywarp_crop = Y_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        X_crop = X_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]
        Y_crop = Y_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]

        # crop grid
        Xwarp_crop_range = \
            2 * (Xwarp_crop - X_crop.min()) / (X_crop.max() - X_crop.min()) - 1
        Ywarp_crop_range = \
            2 * (Ywarp_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min()) - 1
        grid_crop = torch.stack([Xwarp_crop_range,
                                 Ywarp_crop_range], dim=-1)
        return grid_full.unsqueeze(0), grid_crop.unsqueeze(0)

    @staticmethod
    def symmetric_image_pad(image_batch, padding_factor):
        """
        Pad an input image mini-batch symmetrically
        Args:
            image_batch: an input image mini-batch to be pre-processed
            padding_factor: padding factor
        Output:
            image_batch: padded image mini-batch
        """
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))

        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left),
                                 image_batch,
                                 image_batch.index_select(3, idx_pad_right)),
                                3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top),
                                 image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)),
                                2)
        return image_batch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # get the transformation type flag
        transform_type = data['aff/tps/homo'].astype('uint8')
        source_img_name = osp.join(self.img_path, data.fname)
        if not os.path.exists(source_img_name): # If new version of ADE20K
            fname_list = data.fname.split('/')
            fname_list = fname_list[0:3]+[fname_list[-1]] # Remove obsolete path
            data.fname = '/'.join(fname_list)
            source_img_name = osp.join(self.img_path, data.fname)
        if not os.path.exists(source_img_name):
            raise ValueError("The path to one of the original image {} does not exist, check your image path "
                             "and your csv file !".format(source_img_name))
        print('[{}/{}] SOURCE IMG PATH: {}'.format(idx,len(self.df),source_img_name))
        return_dict = dict(transform_type=transform_type)
        return_dict.update({'image': {'ref': {}, 'query': {}},
                            'mask': {'ref': {}, 'query': {}},
                            'flow':{},
                            'corr_mask':{}}
                           )
        # aff/tps transformations
        if transform_type == 0 or transform_type == 1:
            # prepare theta
            theta = self.prepare_theta(data,transform_type)
            theta_extra = self.prepare_extra_theta(transform_type)
            # read image
            source_img = cv2.cvtColor(cv2.imread(source_img_name),
                                      cv2.COLOR_BGR2RGB)
            # cropping dimention of the image first if it is too big, would occur to big resizing after
            if source_img.shape[0] > self.H_AFF_TPS*self.ratio_cropping or \
                    source_img.shape[1] > self.W_AFF_TPS*self.ratio_cropping:
                source_img, x, y = center_crop(source_img, (int(self.W_AFF_TPS*self.ratio_cropping),
                                                            int(self.H_AFF_TPS*self.ratio_cropping)))
            # make arrays float tensor for subsequent processing
            image = torch.Tensor(source_img.astype(np.float32))
            if image.numpy().ndim == 2:
                image = \
                    torch.Tensor(np.dstack((source_img.astype(np.float32),
                                            source_img.astype(np.float32),
                                            source_img.astype(np.float32))))
            image = image.transpose(1, 2).transpose(0, 1)
            # Resize image using bilinear sampling with identity affine
            image_pad = self.transform_image(image.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS)
            # Paste object thumbnail from COCO dataset
            pasted, _ = self.copy_paste(image_pad,idx,num_min_pixel=10000,num_max_pixel=100000)
            image_pad = torch.FloatTensor(pasted['image']).permute(2,0,1) # c,h,w
            flow, correspondence_mask = self.get_flow(transform_type, theta)

            for change_type in ['static','new','missing','replaced','rotated']:
                # prepare both paste-applied & original
                original_img = image_pad
                paste_applied, mask1 = self.copy_paste(original_img, idx, num_min_pixel=10000, num_max_pixel=100000)
                paste_applied = torch.FloatTensor(paste_applied['image']).permute(2, 0, 1)  # c,h,w
                mask1 = torch.FloatTensor(mask1).permute(2, 0, 1) # c, h, w
                paste_applied_with_mask = torch.cat([paste_applied, mask1], dim=0) # 2c, h, w
                original_img_with_mask = torch.cat([original_img,mask1],dim=0) # 2c, h, w
                if change_type == 'static':
                    # Source img
                    img_src_crop = \
                        self.transform_image(original_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16).squeeze()
                    img_src_crop, mask_src_crop = torch.split(img_src_crop,3,dim=0)
                    # Target img
                    img_tgt_crop = \
                        self.transform_image(original_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9/16,
                                             theta=theta).squeeze()
                    img_tgt_crop, mask_tgt_crop = torch.split(img_tgt_crop,3,dim=0)
                    mask_tgt_crop = torch.zeros_like(mask_tgt_crop)

                elif change_type == 'new':
                    # Source img
                    img_src_crop = \
                        self.transform_image(original_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16).squeeze()
                    img_src_crop, mask_src_crop = torch.split(img_src_crop,3,dim=0)
                    # Target img
                    img_tgt_crop = \
                        self.transform_image(paste_applied_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9/16,
                                             theta=theta).squeeze()
                    img_tgt_crop, mask_tgt_crop = torch.split(img_tgt_crop,3,dim=0)

                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                elif change_type == 'missing':
                    # Source img
                    img_src_crop = \
                        self.transform_image(paste_applied_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16).squeeze()
                    img_src_crop, mask_src_crop = torch.split(img_src_crop, 3, dim=0)
                    # Target img
                    img_tgt_crop = \
                        self.transform_image(original_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16,
                                             theta=theta).squeeze()
                    img_tgt_crop, mask_tgt_crop = torch.split(img_tgt_crop, 3, dim=0)

                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                elif change_type == 'replaced':
                    # sample source object thumbnail, then transform it
                    obj_src_dict = self.extract_obj_from_COCO()
                    obj_src_dict = self.rescale_transform(**obj_src_dict)
                    (src_h, src_w, c) = obj_src_dict['image'].shape
                    # sample target object thumbnail, then transform it
                    obj_tgt_dict = self.extract_obj_from_COCO(aspect_ratio_range=[src_h/src_w-0.2,src_h/src_w+0.2])
                    obj_tgt_dict = self.rescale_transform(**obj_tgt_dict)
                    (tgt_h, tgt_w, c) = obj_tgt_dict['image'].shape

                    orig_cx, orig_cy = original_img.shape[2] // 2, original_img.shape[1] // 2
                    x_range = [orig_cx-self.W_OUT//2,orig_cx+self.W_OUT//2-src_w]
                    y_range = [orig_cy-self.W_OUT//2,orig_cy+self.W_OUT//2-src_h]
                    x = random.randint(x_range[0], x_range[1])
                    y = random.randint(y_range[0], y_range[1])
                    src_img = original_img.clone()
                    src_img[:,y:y+src_h,x:x+src_w] = \
                        torch.LongTensor(obj_src_dict['masks'][0])*torch.LongTensor(obj_src_dict['image']).permute(2,0,1) \
                        + torch.LongTensor(1-obj_src_dict['masks'][0])*original_img[:,y:y+src_h,x:x+src_w]
                    src_mask = torch.zeros(original_img.shape[1:])
                    src_mask[y:y+src_h,x:x+src_w] = torch.LongTensor(obj_src_dict['masks'][0])

                    tgt_img = original_img.clone()
                    tgt_img[:,y:y+tgt_h,x:x+tgt_w] = \
                        torch.LongTensor(obj_tgt_dict['masks'][0])*torch.LongTensor(obj_tgt_dict['image']).permute(2,0,1) \
                        + torch.LongTensor(1-obj_tgt_dict['masks'][0])*original_img[:,y:y+tgt_h,x:x+tgt_w]
                    tgt_mask = torch.zeros(original_img.shape[1:])
                    tgt_mask[y:y+tgt_h,x:x+tgt_w] = torch.LongTensor(obj_tgt_dict['masks'][0])
                    src_img_with_mask = torch.cat([src_img,src_mask[None,...]], dim=0)
                    tgt_img_with_mask = torch.cat([tgt_img, tgt_mask[None,...]], dim=0)

                    # Source img
                    img_src_crop = \
                        self.transform_image(src_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16).squeeze()
                    #img_src_crop, mask_src_crop = torch.split(img_src_crop, 3, dim=0)
                    img_src_crop, mask_src_crop = img_src_crop[:3], img_src_crop[3]

                    # Target img
                    img_tgt_crop = \
                        self.transform_image(tgt_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16,
                                             theta=theta).squeeze()
                    #img_tgt_crop, mask_tgt_crop = torch.split(img_tgt_crop, 3, dim=0)
                    img_tgt_crop, mask_tgt_crop = img_tgt_crop[:3], img_tgt_crop[3].long()
                    mask_tgt_crop = self.inFill(mask_tgt_crop) # convert (h,w) into (h,w,3)
                    mask_src_crop = self.inFill(mask_src_crop) # convert (h,w) into (h,w,3)

                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                elif change_type == 'rotated':
                    obj_src_dict = self.extract_obj_from_COCO()
                    obj_src_dict = self.rescale_transform(**obj_src_dict)
                    (src_h, src_w, c) = obj_src_dict['image'].shape
                    obj_img_with_mask = torch.cat([
                        torch.FloatTensor(obj_src_dict['image']).permute(2,0,1),
                        torch.FloatTensor(obj_src_dict['masks'][0])[None,...]],dim=0)
                    # Source img
                    img_src_crop = \
                        self.transform_image(obj_img_with_mask[None, ...],
                                             src_h,
                                             src_w,
                                             padding_factor=1.1,
                                             crop_factor=1,
                                             theta=theta).squeeze()
                    img_src_crop = self.rescale_transform(image=img_src_crop.permute(1,2,0).numpy())['image']
                    img_src_crop, mask_src_crop = img_src_crop[:,:,:3], img_src_crop[:,:,3]
                    # img_src_crop, mask_src_crop = torch.split(img_src_crop, 3, dim=0)

                    img_tgt_crop = \
                        self.transform_image(obj_img_with_mask[None, ...],
                                             src_h,
                                             src_w,
                                             padding_factor=1.1,
                                             crop_factor=1,
                                             theta=theta_extra).squeeze()

                    img_tgt_crop = self.rotate_transform(image=img_tgt_crop.permute(1,2,0).numpy())['image']
                    img_tgt_crop = self.rescale_transform(image=img_tgt_crop)['image']
                    img_tgt_crop, mask_tgt_crop = img_tgt_crop[:,:,:3], img_tgt_crop[:,:,3]

                    obj_src_dict = dict(image=img_src_crop,masks=[mask_src_crop])
                    obj_tgt_dict = dict(image=img_tgt_crop,masks=[mask_tgt_crop])
                    src_img_with_mask,xy = self.paste_from_cropped_obj(original_img, obj_src_dict)
                    obj_img_with_mask,xy = self.paste_from_cropped_obj(torch.zeros_like(original_img),obj_src_dict,xy)

                    #src_img_with_mask, tgt_img_with_mask = self.paste_from_cropped_objs(original_img, obj_src_dict, obj_tgt_dict)
                    # obj img
                    obj_tgt_crop = \
                        self.transform_image(obj_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16,
                                             theta=theta_extra).squeeze()
                    obj_tgt_crop, mask_tgt_crop = obj_tgt_crop[:3], obj_tgt_crop[3].long()

                    # Source img
                    img_src_crop = \
                        self.transform_image(src_img_with_mask[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16).squeeze()
                    img_src_crop, mask_src_crop = img_src_crop[:3], img_src_crop[3].long()

                    # Target img
                    img_tgt_crop = \
                        self.transform_image(original_img[None,...],
                                             self.H_OUT,
                                             self.W_OUT,
                                             padding_factor=0.8,
                                             crop_factor=9 / 16,
                                             theta=theta).squeeze()
                    img_tgt_crop = torch.where(mask_tgt_crop.bool(),obj_tgt_crop,img_tgt_crop)
                    #img_tgt_crop, mask_tgt_crop = img_tgt_crop[:3], img_tgt_crop[3]
                    obj_flow, _ = self.get_flow(transform_type, theta_extra)
                    mask_tgt_crop = self.inFill(mask_tgt_crop) # convert (h,w) into (h,w,3)
                    mask_src_crop = self.inFill(mask_src_crop) # convert (h,w) into (h,w,3)
                    flow = torch.where(torch.BoolTensor(mask_tgt_crop[:,:,0]),obj_flow,flow)

                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                else:
                    raise ValueError

                return_dict['flow'][change_type] = flow
                return_dict['image']['ref'][change_type] = self.to_tensor_hwc_format(img_src_crop)
                return_dict['image']['query'][change_type] = self.to_tensor_hwc_format(img_tgt_crop)
                return_dict['mask']['ref'][change_type] = self.to_tensor_hwc_format(mask_src_crop)
                return_dict['mask']['query'][change_type] = self.to_tensor_hwc_format(mask_tgt_crop)

        # Homography transformation
        elif transform_type == 2:
            # ATTENTION CV2 resize is inverted, first w and then h
            theta = self.prepare_theta(data,transform_type)
            theta_extra = self.prepare_extra_theta(transform_type)
            source_img = cv2.cvtColor(cv2.imread(source_img_name), cv2.COLOR_BGR2RGB)
            # cropping dimention of the image first if it is too big, would occur to big resizing after
            if source_img.shape[0] > self.H_HOMO * self.ratio_cropping \
                    or source_img.shape[1] > self.W_HOMO*self.ratio_cropping:
                source_img, x, y = center_crop(source_img, (int(self.W_HOMO*self.ratio_cropping),
                                                            int(self.H_HOMO*self.ratio_cropping)))
            # resize to value stated at the beginning
            img_src_orig = cv2.resize(source_img, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR) # cv2.resize, W is giving first
            # Paste object thumbnail from COCO dataset
            img_src_orig, _ = self.copy_paste(img_src_orig,idx,num_min_pixel=10000,num_max_pixel=100000)
            img_src_orig = img_src_orig['image']
            # get a central crop:
            img_src_crop, x1_crop, y1_crop = center_crop(img_src_orig, self.W_OUT)
            # Obtaining the full and crop grids out of H
            grid_full, grid_crop = self.get_grid(theta, ccrop=(x1_crop, y1_crop))

            for change_type in ['static','new','missing','replaced','rotated']:
                # prepare both paste-applied & original
                # img_src_crop = torch.FloatTensor(img_src_crop).permute(2,0,1) # c,h,w
                paste_applied, mask_src_crop = self.copy_paste(img_src_crop, idx, num_min_pixel=10000, num_max_pixel=100000)

                img_src_orig_pasted = img_src_orig.copy() # h,w,c
                mask_src_orig_pasted = np.zeros_like(img_src_orig_pasted) # h,w,c
                img_src_orig_pasted[y1_crop:y1_crop+self.W_OUT,x1_crop:x1_crop+self.W_OUT] = paste_applied['image']
                mask_src_orig_pasted[y1_crop:y1_crop+self.W_OUT,x1_crop:x1_crop+self.W_OUT] = mask_src_crop

                img_src_orig_pasted = torch.FloatTensor(img_src_orig_pasted).permute(2, 0, 1)  # c,h,w
                mask_src_orig_pasted = torch.FloatTensor(mask_src_orig_pasted).permute(2, 0, 1) # c, h, w
                paste_applied_with_mask = torch.cat([img_src_orig_pasted, mask_src_orig_pasted], dim=0) # 2c, h, w
                original_img_with_mask = torch.cat([torch.FloatTensor(img_src_orig).permute(2,0,1),
                                                    mask_src_orig_pasted], dim=0) # 2c, h, w
                if transform_type in (0, 1): grid_crop = None
                flow, correspondence_mask = self.get_flow(transform_type, theta, grid_crop)

                if change_type =='static':
                    img_src_crop = img_src_crop.copy()
                    mask_src_crop = np.zeros_like(img_src_crop) # dummy mask (empty)
                    img_orig_tgt_vrbl_with_mask, _ = self.Homography_transform(img_src_orig,
                                                                               theta, x1_crop, y1_crop, mask=None) # 2c,h,w
                    img_tgt_crop_with_mask, _, _ = center_crop(img_orig_tgt_vrbl_with_mask.permute(1,2,0).numpy(),
                                                               self.W_OUT) # 520,520,2c
                    img_tgt_crop, mask_tgt_crop = img_tgt_crop_with_mask[:,:,:3], np.zeros_like(mask_src_crop) # dummy mask (empty)

                elif change_type == 'new':
                    img_src_crop = img_src_crop.copy()
                    img_orig_tgt_vrbl_with_mask, _ = self.Homography_transform(paste_applied_with_mask,
                                                                               theta, x1_crop, y1_crop, mask=None) # 2c,h,w
                    img_tgt_crop_with_mask, _, _ = center_crop(img_orig_tgt_vrbl_with_mask.permute(1,2,0).numpy(),
                                                               self.W_OUT) # 520,520,2c
                    img_tgt_crop, mask_tgt_crop = img_tgt_crop_with_mask[:,:,:3], img_tgt_crop_with_mask[:,:,3:]

                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                elif change_type == 'missing':
                    img_src_crop = paste_applied['image'].copy()
                    img_orig_tgt_vrbl_with_mask, _ = self.Homography_transform(original_img_with_mask,
                                                                               theta, x1_crop, y1_crop, mask=None) # 2c,h,w
                    img_tgt_crop_with_mask, _, _ = center_crop(img_orig_tgt_vrbl_with_mask.permute(1,2,0).numpy(),
                                                               self.W_OUT) # 520,520,2c
                    img_tgt_crop, mask_tgt_crop = img_tgt_crop_with_mask[:,:,:3], img_tgt_crop_with_mask[:,:,3:]

                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                elif change_type == 'replaced':
                    # sample source object thumbnail, then rescale it
                    obj_src_dict = self.extract_obj_from_COCO()
                    obj_src_dict = self.rescale_transform(**obj_src_dict)
                    (src_h, src_w, c) = obj_src_dict['image'].shape
                    # sample target object thumbnail, then rescale it
                    obj_tgt_dict = self.extract_obj_from_COCO(aspect_ratio_range=[src_h/src_w-0.2,src_h/src_w+0.2])
                    obj_tgt_dict = self.rescale_transform(**obj_tgt_dict)
                    (tgt_h, tgt_w, c) = obj_tgt_dict['image'].shape

                    src_img_with_mask, tgt_img_with_mask = self.paste_from_cropped_objs(img_src_orig, obj_src_dict, obj_tgt_dict)

                    img_orig_tgt_vrbl_with_mask, _ = self.Homography_transform(tgt_img_with_mask,
                                                                               theta, x1_crop, y1_crop, mask=None) # 2c,h,w
                    img_tgt_crop_with_mask, _, _ = center_crop(img_orig_tgt_vrbl_with_mask.permute(1,2,0).numpy(),
                                                               self.W_OUT) # 520,520,c+1
                    img_src_crop_with_mask, _, _ = center_crop(src_img_with_mask.permute(1,2,0).numpy(),
                                                               self.W_OUT) # 520,520,c+1
                    img_src_crop, mask_src_crop = img_src_crop_with_mask[:,:,:3], img_src_crop_with_mask[:,:,3]
                    img_tgt_crop, mask_tgt_crop = img_tgt_crop_with_mask[:,:,:3], img_tgt_crop_with_mask[:,:,3]

                    mask_tgt_crop = self.inFill(mask_tgt_crop) # convert (h,w) into (h,w,3)
                    mask_src_crop = self.inFill(mask_src_crop) # convert (h,w) into (h,w,3)


                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                elif change_type == 'rotated':
                    obj_src_dict = self.extract_obj_from_COCO()
                    obj_src_dict = self.rescale_transform(**obj_src_dict)
                    (src_h, src_w, c) = obj_src_dict['image'].shape
                    obj_img_with_mask = torch.cat([
                        torch.FloatTensor(obj_src_dict['image']).permute(2,0,1),
                        torch.FloatTensor(obj_src_dict['masks'][0])[None,...]],dim=0)
                    obj_img, obj_mask = obj_img_with_mask[:,:,:3], obj_img_with_mask[:,:,3]

                    # rotated obj thumbnail
                    transform_type_extra = random.randint(0,1)
                    theta_extra = self.prepare_extra_theta(transform_type=transform_type_extra)
                    src_obj_img_with_mask, xy = self.paste_from_cropped_obj(img_src_orig,obj_dict=obj_src_dict)
                    # src_obj_img_with_mask, _ = self.Homography_transform(src_obj_img_with_mask,
                    #                                                            theta, x1_crop, y1_crop,
                    #                                                            mask=None)  # 2c,h,w
                    img_src_crop_with_mask, _, _ = center_crop(src_obj_img_with_mask.permute(1, 2, 0).numpy(),
                                                               self.W_OUT)  # 520,520,c+1
                    bg_img = np.zeros([self.H_OUT, self.H_OUT, 3])
                    bg_obj_img_with_mask, _ = self.paste_from_cropped_obj(bg_img,obj_dict=obj_src_dict,
                                                                          xy=(xy[0]-x1_crop,xy[1]-y1_crop))
                    bg_obj_img_with_mask = \
                        self.transform_image(bg_obj_img_with_mask[None, ...],
                                             bg_obj_img_with_mask.shape[1],
                                             bg_obj_img_with_mask.shape[2],
                                             padding_factor=1.0,
                                             crop_factor=1,
                                             theta=theta_extra).squeeze()  # 4,520,520 empty except object-containing pixels



                    img_orig_tgt_vrbl_with_mask, _ = self.Homography_transform(img_src_orig,
                                                                               theta, x1_crop, y1_crop, mask=None) # 2c,h,w
                    img_tgt_crop_with_mask, _, _ = center_crop(img_orig_tgt_vrbl_with_mask.permute(1,2,0).numpy(),
                                                               self.W_OUT) # 520,520,2c
                    img_tgt_crop = img_tgt_crop_with_mask[:,:,:3]
                    img_tgt_crop = torch.where(bg_obj_img_with_mask[3].bool(),bg_obj_img_with_mask[:3],
                                               torch.FloatTensor(img_tgt_crop).permute(2,0,1))
                    mask_tgt_crop = bg_obj_img_with_mask[3]
                    img_src_crop, mask_src_crop = img_src_crop_with_mask[:,:,:3], img_src_crop_with_mask[:,:,3]
                    mask_tgt_crop = self.inFill(mask_tgt_crop) # convert (h,w) into (h,w,3)
                    mask_src_crop = self.inFill(mask_src_crop) # convert (h,w) into (h,w,3)

                    bg_flow, _ = self.get_flow(transform_type,theta,grid_crop)
                    obj_flow,_ = self.get_flow(transform_type_extra,theta_extra)
                    flow = torch.where(torch.BoolTensor(mask_tgt_crop[:,:,0]),obj_flow,bg_flow)

                    #img_tgt_crop, mask_tgt_crop = img_tgt_crop_with_mask[:,:,:3], img_tgt_crop_with_mask[:,:,3]



                    self.viz_change_aug(img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                                        change_type,transform_type,idx,show=False)

                else:
                    raise ValueError

                return_dict['flow'][change_type] = flow
                return_dict['image']['ref'][change_type] = self.to_tensor_hwc_format(img_src_crop)
                return_dict['image']['query'][change_type] = self.to_tensor_hwc_format(img_tgt_crop)
                return_dict['mask']['ref'][change_type] = self.to_tensor_hwc_format(mask_src_crop)
                return_dict['mask']['query'][change_type] = self.to_tensor_hwc_format(mask_tgt_crop)

        else:
            print('Error: transformation type')
            raise ValueError

        # # if transform_type in (0,1): grid_crop = None
        # # flow, correspondence_mask = self.get_flow(transform_type,theta,grid_crop)
        # return_dict.update(correspondence_mask=correspondence_mask)

        return return_dict

    def apply_object_flow(self, flow_static, src_obj_mask, transform_type_extra, theta_extra):
        '''
        flow_static: 2,h,w
        src_obj_mask: h,w
        '''
        flow_obj, correspondence_mask_obj = self.get_flow(transform_type_extra, theta_extra)
        flow_with_dynamic_obj = flow_static.clone()  # 2,h,w
        flow_c, flow_h, flow_w = flow_with_dynamic_obj.shape
        src_obj_mask = src_obj_mask[None, ...].repeat(flow_c, 1, 1).long()  # 2,h,w
        flow_with_dynamic_obj = flow_with_dynamic_obj.contiguous().view(-1)  # flatten h and w
        flow_with_dynamic_obj[src_obj_mask.contiguous().view(-1).nonzero()] = \
            flow_obj.contiguous().view(-1)[src_obj_mask.contiguous().view(-1).nonzero()]
        flow_with_dynamic_obj = flow_with_dynamic_obj.view(flow_c, flow_h, flow_w)
        return flow_with_dynamic_obj

    def get_flow(self,transform_type, theta, grid_crop = None,pyramid_param=None, H_OUT=None, W_OUT=None):
        # construct a pyramid with a corresponding grid on each layer
        grid_pyramid = []
        mask_x = []
        mask_y = []
        if pyramid_param is None: pyramid_param = self.pyramid_param
        if transform_type == 0:
            for layer_size in pyramid_param:
                # get layer size or change it so that it corresponds to PWCNet
                grid = self.generate_grid(layer_size,
                                          layer_size,
                                          theta).squeeze(0)
                mask = grid.ge(-1) & grid.le(1)
                grid_pyramid.append(grid)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])
        elif transform_type == 1:
            grid = self.generate_grid(self.H_OUT,
                                      self.W_OUT,
                                      theta).squeeze(0)
            for layer_size in pyramid_param:
                grid_m = torch.from_numpy(cv2.resize(grid.numpy(),
                                                     (layer_size, layer_size)))
                mask = grid_m.ge(-1) & grid_m.le(1)
                grid_pyramid.append(grid_m)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])
        elif transform_type == 2:
            assert grid_crop is not None
            grid = grid_crop.squeeze(0)
            for layer_size in pyramid_param:
                grid_m = torch.from_numpy(cv2.resize(grid.numpy(),
                                                    (layer_size, layer_size)))
                mask = grid_m.ge(-1) & grid_m.le(1)
                grid_pyramid.append(grid_m)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])

        # ATTENTION, here we just get the flow of the highest resolution asked, not the pyramid of flows !
        flow = unormalise_and_convert_mapping_to_flow(grid_pyramid[-1], output_channel_first=True) # 2,h,w
        mask = torch.logical_and(mask_x[-1], mask_y[-1])

        return flow, mask

    def copy_paste(self,img_src_orig,idx,num_min_pixel=10000,num_max_pixel=100000,target_bbox=None):
        '''
        params:
            img_src_orig: torch tensor of size (c,h,w) or numpy array of size (h,w,c)
        return:
            pasted: dict containing image, bboxes, masks.
            mask: binary mask
        '''

        if isinstance(img_src_orig,(torch.LongTensor,torch.FloatTensor)):
            img_src_orig = img_src_orig.squeeze() # single-image only supported.
            assert img_src_orig.shape.__len__() == 3
            c,h,w = img_src_orig.shape
            assert c == 3
            img_src_orig = img_src_orig.permute(1,2,0).numpy().astype('uint8') # channel_dim last
        elif isinstance(img_src_orig, np.ndarray):
            h,w,c = img_src_orig.shape
            assert c ==3
        else:
            raise NotImplementedError


        while 1:
            try:
                # Load image from COCO dataset
                coco_data = self.coco_data.load_example(index=random.randint(0, self.coco_data.ids.__len__()))
                (h, w, c) = coco_data['image'].shape

                # Resize & Padding
                min_crop_length = min(h, w)
                min_resize_length = min(img_src_orig.shape[0], img_src_orig.shape[1])
                tf_crop_resize = A.Compose([
                    A.CenterCrop(height=min_crop_length, width=min_crop_length),
                    A.Resize(height=self.W_OUT, width=self.W_OUT),
                    A.PadIfNeeded(min_height=img_src_orig.shape[0], min_width=img_src_orig.shape[1],
                                  border_mode=cv2.BORDER_CONSTANT, value=(0, 255, 0))
                ])
                #import pdb;pdb.set_trace()
                coco_crop = tf_crop_resize(**coco_data)

                # Filter out too big/small masks
                coco_crop, num_bbox = self.filter_by_size_and_ratio(**coco_crop)
                if num_bbox == 0:
                    print("[{}/{}] No mask remaining. trying new one...".format(idx, self.__len__()))
                    continue

                # Do copy-paste
                pasted = self.copy_paste_transform_prob100(image=img_src_orig,
                                                          masks=[np.zeros_like(coco_crop['image'][:, :, 0])],
                                                          bboxes=[[0.0, 0.0, 1.0, 1.0, 0]],
                                                          paste_image=coco_crop['image'], paste_masks=coco_crop['masks'],
                                                          paste_bboxes=coco_crop['bboxes'])
                mask = sum(pasted['masks']).astype('bool').astype('uint8')
            except Exception as e:
                print(e)
                print("[{}/{}] error occurred. trying new one...".format(idx, self.__len__()))
                continue
            if np.sum(mask) < num_min_pixel:
                print("[{}/{}] mask too small. trying new one...".format(idx, self.__len__()))
                continue
            elif np.sum(mask) > num_max_pixel:
                print("[{}/{}] mask too big. trying new one...".format(idx, self.__len__()))
                continue
            else:
                # proper mask !
                print("[{}/{}] copy&paste successful".format(idx, self.__len__()))
                mask = np.tile(mask[:, :, None], (1, 1, 3)) * 255  # grayscale [0,1] to RGB [0,255]
                break

        return pasted, mask

    def filter_by_size_and_ratio(self, image, bboxes, masks, num_pixel_range=[10000, 70000],
                                 aspect_ratio_range=[0.5,2]):
        # Filter by area
        proper_mask_idx = [i for i, m in enumerate(masks) if
                           m.sum() < num_pixel_range[1] and m.sum() > num_pixel_range[0]]
        masks = [m for i, m in enumerate(masks) if i in proper_mask_idx]
        bboxes = [bb for i, bb in enumerate(bboxes) if i in proper_mask_idx]

        # Filter by bbox aspect ratio
        proper_bbox_idx = [i for i, bb in enumerate(bboxes)
                           if bb[3]/bb[2] <aspect_ratio_range[1] and bb[3]/bb[2]>aspect_ratio_range[0]]
        masks = [m for i, m in enumerate(masks) if i in proper_bbox_idx]
        bboxes = [bb for i, bb in enumerate(bboxes) if i in proper_bbox_idx]

        bboxes = [tuple(list(bb[:-1]) + [i]) for i, bb in
                               enumerate(bboxes)]  # rearrange bbbox idxs
        return dict(image=image,bboxes=bboxes,masks=masks), len(bboxes)

    def extract_obj_from_COCO(self,aspect_ratio_range=[0.5,2.0]):
        num_proper_bbox=0
        while num_proper_bbox==0:
            # Load image from COCO dataset
            coco_data = self.coco_data.load_example(index=random.randint(0, self.coco_data.ids.__len__()))
            (h, w, c) = coco_data['image'].shape
            coco_data, num_proper_bbox = self.filter_by_size_and_ratio(**coco_data, aspect_ratio_range=aspect_ratio_range)

        sample_idx = random.randint(0,len(coco_data['bboxes'])-1)
        bboxes = [coco_data['bboxes'][sample_idx]]
        bboxes = [tuple(list(bbox[:-1])+[i]) for i, bbox in enumerate(bboxes)]
        src_data = dict(image=coco_data['image'],bboxes=bboxes,
                        masks=[coco_data['masks'][sample_idx]])
        background_data = dict(image=np.zeros_like(coco_data['image']),
                          bboxes=[[0,0,1,1,0]],masks=[np.zeros_like(coco_data['image'][:, :, 0])])
        background_data.update(paste_image=src_data['image'],
                               paste_masks=src_data['masks'],
                               paste_bboxes=src_data['bboxes'])
        pasted = self.copy_paste_transform_prob100(**background_data)
        pasted['masks'].pop(0)
        xydxdy = pasted['bboxes'][0]

        # crop
        img_crop = pasted['image'][int(xydxdy[1]):int(xydxdy[1]+xydxdy[3]),int(xydxdy[0]):int(xydxdy[0]+xydxdy[2])]
        mask_crop = pasted['masks'][0][int(xydxdy[1]):int(xydxdy[1]+xydxdy[3]),int(xydxdy[0]):int(xydxdy[0]+xydxdy[2])]

        '''DEBUG'''
        # plt.close()
        # plt.subplot(311)
        # plt.imshow(img_crop.astype('uint8'))
        # plt.subplot(312)
        # plt.imshow(pasted['image'].astype('uint8'))
        # plt.subplot(313)
        # plt.imshow(pasted['masks'][0].astype('uint8'))
        # plt.show()

        return dict(image=img_crop,masks=[mask_crop],bboxes=[[0,0,img_crop.shape[1]-1,img_crop.shape[0]-1,0,0]])

    def paste_from_cropped_objs(self, original_img, obj_src_dict, obj_tgt_dict):

        if isinstance(original_img,np.ndarray):
            original_img = torch.FloatTensor(original_img).permute(2,0,1)

        (src_h, src_w, c) = obj_src_dict['image'].shape
        (tgt_h, tgt_w, c) = obj_tgt_dict['image'].shape
        orig_cx, orig_cy = original_img.shape[2] // 2, original_img.shape[1] // 2
        x_range = [orig_cx - self.H_OUT // 2, orig_cx + self.H_OUT // 2 - src_w]
        y_range = [orig_cy - self.H_OUT // 2, orig_cy + self.H_OUT // 2 - src_h]
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        src_img = original_img.clone()
        src_img[:, y:y + src_h, x:x + src_w] = \
            torch.LongTensor(obj_src_dict['masks'][0]) * torch.LongTensor(obj_src_dict['image']).permute(2, 0, 1) \
            + torch.LongTensor(1 - obj_src_dict['masks'][0]) * original_img[:, y:y + src_h, x:x + src_w]
        src_mask = torch.zeros(original_img.shape[1:])
        src_mask[y:y + src_h, x:x + src_w] = torch.LongTensor(obj_src_dict['masks'][0])
        tgt_img = original_img.clone()
        tgt_img[:, y:y + tgt_h, x:x + tgt_w] = \
            torch.LongTensor(obj_tgt_dict['masks'][0]) * torch.LongTensor(obj_tgt_dict['image']).permute(2, 0, 1) \
            + torch.LongTensor(1 - obj_tgt_dict['masks'][0]) * original_img[:, y:y + tgt_h, x:x + tgt_w]
        tgt_mask = torch.zeros(original_img.shape[1:])
        tgt_mask[y:y + tgt_h, x:x + tgt_w] = torch.LongTensor(obj_tgt_dict['masks'][0])
        src_img_with_mask = torch.cat([src_img, src_mask[None, ...]], dim=0)
        tgt_img_with_mask = torch.cat([tgt_img, tgt_mask[None, ...]], dim=0)

        return src_img_with_mask, tgt_img_with_mask

    def paste_from_cropped_obj(self, bg_img, obj_dict,xy=None):

        if isinstance(bg_img, np.ndarray):
            bg_img = torch.FloatTensor(bg_img).permute(2, 0, 1)

        (src_h, src_w, c) = obj_dict['image'].shape
        orig_cx, orig_cy = bg_img.shape[2] // 2, bg_img.shape[1] // 2
        x_range = [orig_cx - self.H_OUT // 2, orig_cx + self.H_OUT // 2 - src_w]
        y_range = [orig_cy - self.H_OUT // 2, orig_cy + self.H_OUT // 2 - src_h]
        if xy is not None:
            x,y = xy
        else:
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(y_range[0], y_range[1])

        src_img = bg_img.clone()
        src_img[:, y:y + src_h, x:x + src_w] = \
            torch.LongTensor(obj_dict['masks'][0]) * torch.LongTensor(obj_dict['image']).permute(2, 0, 1) \
            + torch.LongTensor(1 - obj_dict['masks'][0]) * bg_img[:, y:y + src_h, x:x + src_w]
        src_mask = torch.zeros(bg_img.shape[1:])
        src_mask[y:y + src_h, x:x + src_w] = torch.LongTensor(obj_dict['masks'][0])

        src_img_with_mask = torch.cat([src_img, src_mask[None, ...]], dim=0)

        return src_img_with_mask, (x,y)

    def Homography_transform(self,img_src_orig,theta,x1_crop,y1_crop, mask=None):

        # Obtaining the full and crop grids out of H
        grid_full, grid_crop = self.get_grid(theta,
                                             ccrop=(x1_crop, y1_crop))

        if isinstance(img_src_orig,np.ndarray):
            assert len(img_src_orig.shape)==3 and img_src_orig.shape[2]==3
            img_src_orig = torch.Tensor(img_src_orig.astype(np.float32))
            img_src_orig = img_src_orig.permute(2, 0, 1)
        if isinstance(img_src_orig,torch.Tensor):
            pass
        else:
            raise ValueError

        # warp the fullsize original source image
        if mask is not None:
            mask = torch.Tensor(mask.astype(np.float32))
            mask = mask.permute(2, 0, 1)
        if float(torch.__version__[:3]) >= 1.3:
            img_orig_target_vrbl = F.grid_sample(img_src_orig.unsqueeze(0),
                                                 grid_full, align_corners=True)
            if mask is not None:
                mask_orig_target_vrbl = F.grid_sample(mask.unsqueeze(0),
                                                      grid_full, align_corners=True).squeeze()
            else:
                mask_orig_target_vrbl = None
        else:
            img_orig_target_vrbl = F.grid_sample(img_src_orig.unsqueeze(0),
                                                 grid_full)
            if mask is not None:
                mask_orig_target_vrbl = F.grid_sample(mask.unsqueeze(0),
                                                      grid_full).squeeze()
            else:
                mask_orig_target_vrbl = None

        return img_orig_target_vrbl.squeeze(), mask_orig_target_vrbl

    def viz_change_aug(self,img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop,
                       change_type,transform_type,idx,show=False,save=False):
        if show + save > 0:
            plot_idx = 141
            plt.clf()
            plt.close()
            plt.figure(figsize=(12, 3))
            plt.suptitle('{}_{}_{}'.format(idx,change_type,transform_type))
            for img in [img_src_crop,img_tgt_crop,mask_src_crop,mask_tgt_crop]:
                if isinstance(img,torch.Tensor):
                    if len(img.shape) ==3:
                        img = img.permute(1,2,0).numpy()
                    elif len(img.shape) ==2:
                        img = img.numpy()
                    else:
                        raise ValueError
                ax = plt.subplot(plot_idx)
                plt.imshow(img.astype('uint8'))
                plot_idx+=1

            if show:
                plt.show()
            if save:
                plt.savefig('visualize/{}/{}.png'.format(change_type, idx))

        #

    def prepare_theta(self,data,transform_type):
        # prepare theta
        if transform_type == 0:
            theta = data.iloc[2:8].values.astype('float').reshape(2, 3)
            theta = torch.Tensor(theta.astype(np.float32)).expand(1, 2, 3)

        elif transform_type == 1:
            theta = data.iloc[2:].values.astype('float')
            theta = np.expand_dims(np.expand_dims(theta, 1), 2)
            theta = torch.Tensor(theta.astype(np.float32))
            theta = theta.expand(1, 18, 1, 1)

        elif transform_type == 2:
            # ATTENTION CV2 resize is inverted, first w and then h
            theta = data.iloc[2:11].values.astype('double').reshape(3, 3)

        else:
            raise ValueError

        return theta

    def prepare_extra_theta(self,transform_type):
        if transform_type == 0:
            Flag = -1
            while Flag != transform_type:
                data_extra = self.df.iloc[random.randint(0, self.__len__() - 1)]
                Flag = data_extra['aff/tps/homo'].astype('uint8')
            theta_extra = data_extra.iloc[2:8].values.astype('float').reshape(2, 3)
            theta_extra = torch.Tensor(theta_extra.astype(np.float32)).expand(1, 2, 3)

        elif transform_type == 1:
            Flag = -1
            while Flag != transform_type:
                data_extra = self.df.iloc[random.randint(0, self.__len__() - 1)]
                Flag = data_extra['aff/tps/homo'].astype('uint8')
            theta_extra = data_extra.iloc[2:].values.astype('float')
            theta_extra = np.expand_dims(np.expand_dims(theta_extra, 1), 2)
            theta_extra = torch.Tensor(theta_extra.astype(np.float32))
            theta_extra = theta_extra.expand(1, 18, 1, 1)

        elif transform_type == 2:
            Flag = -1
            while Flag !=2:
                data_extra = self.df.iloc[random.randint(0, self.__len__() - 1)]
                Flag = data_extra['aff/tps/homo'].astype('uint8')
            # ATTENTION CV2 resize is inverted, first w and then h
            theta_extra = data_extra.iloc[2:11].values.astype('double').reshape(3, 3)

        else:
            raise ValueError

        return theta_extra

    def inFill(self,binary_img):
        assert len(binary_img.shape) in (2,3)
        if isinstance(binary_img,torch.Tensor):
            if len(binary_img.shape) == 2:
                binary_img = 255 * binary_img[...,None].repeat(1,1,3)
            binary_img = binary_img.numpy()
        elif isinstance(binary_img,np.ndarray):
            if len(binary_img.shape) == 2:
                binary_img = 255 * binary_img[...,None].repeat(3,axis=2)
        else:
            raise ValueError
        binary_img_original = binary_img.astype('uint8')
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        hh, ww = binary_img_original.shape[:2]
        mask = np.zeros((hh + 2, ww + 2), np.uint8)
        binary_img_filled = binary_img_original.copy()
        # Floodfill from point (0, 0)
        cv2.floodFill(binary_img_filled, mask, (0,0), (255,255,255))
        # Invert floodfilled image
        binary_img_inv = cv2.bitwise_not(binary_img_filled)
        # Combine the two images to get the foreground.
        im_out = binary_img_original | binary_img_inv

        return im_out

    def to_tensor_hwc_format(self,img):
        assert isinstance(img,np.ndarray) or isinstance(img,torch.Tensor)
        if len(img.shape) == 4:
            assert img.shape[0] == 1
            img = img.squeeze()
        assert len(img.shape) in (2,3)
        if isinstance(img,np.ndarray): img = torch.FloatTensor(img)
        if len(img.shape) == 2:
            img = 255 * img[...,None].repeat(1,1,3)
        if len(img.shape) == 3:
            if img.shape[-1] < 4: pass
            else: img = img.permute(1,2,0)

        return img.long()

class TpsGridGen(nn.Module):
    """
    Adopted version of synthetically transformed pairs dataset by I.Rocco
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=False):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)
