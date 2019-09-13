from torch.utils.data import Dataset
import os
import imageio
import glob
from PIL import Image
import numpy as np
from imageio import imread
import scipy.io as io
import h5py
from representations import *
import data_transforms as transforms
import random
import cv2
from api import utils as api_utils
import torch.nn as nn
from imgaug import augmenters as iaa
import imgaug as ia
import torchvision

class GeoDataset(Dataset):
    def __init__(self, img_list, root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_normals=True,
                 input_type='image'):

        if root_dir == '':
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir
        self.transforms = transforms

        self.img_list = img_list
        self.img_size = img_size
        self.use_boundary = use_boundary
        self.use_depth = use_depth
        self.use_normals = use_normals
        self.input_type = input_type

    def __len__(self):
        return len(self.img_list)

    def format_data(self, image=None,
                    mask_valid=None,
                    depth=None,
                    normals=None,
                    boundary=None):
        # augment data and format it to tensor type

        data = [image, mask_valid,
                depth if self.use_depth else None,
                normals if self.use_normals else None,
                boundary if self.use_boundary else None]

        if self.transforms is not None:
            ratio = 1
            crop_size = None
            angle = 0
            gamma_ratio = 1
            normalize = False
            flip = None

            if 'RESIZE_TRANSPARENT' in self.transforms.keys():
                h, w = 256, 256
            if 'SCALE' in self.transforms.keys():
                ratio = random.uniform(1.0 / self.transforms['SCALE'], 1.0 * self.transforms['SCALE'])
            if 'HORIZONTALFLIP' in self.transforms.keys():
                flip = random.random() < 0.5
            if 'CROP' in self.transforms.keys():
                crop_size = self.transforms['CROP']
                # x1, y1, tw, th = transforms.get_random_bbox(data, crop_size, crop_size)
            if 'ROTATE' in self.transforms.keys():
                angle = random.uniform(0, self.transforms['ROTATE'] * 2) - self.transforms['ROTATE']
            if 'GAMMA' in self.transforms.keys():
                gamma_ratio = random.uniform(1 / self.transforms['GAMMA'], self.transforms['GAMMA'])
            if 'NORMALIZE' in self.transforms.keys():
                normalize = True

            for m, mode in enumerate(data):
                if mode is not None:
                    if m == 0:
                        mode.resize(w, h, cv2.INTER_LINEAR)
                    else:
                        mode.resize(w, h, cv2.INTER_NEAREST)

            for mode in data:
                if mode is not None:
                    if ratio != 1:
                        mode.scale(ratio)
            if crop_size is not None:
                data = transforms.get_random_crop(data, crop_size, crop_size)
            for m, mode in enumerate(data):
                if mode is not None:
                    if flip:
                        mode.fliplr()
                    if angle != 0:
                        mode.rotate(angle, cval=0)
                    if m == 0:
                        if gamma_ratio != 1:
                            data[0].gamma(gamma_ratio)
                    mode.to_tensor()
                    if m == 0:
                        mode.normalize(mean=self.transforms['NORMALIZE']['mean'],
                                       std=self.transforms['NORMALIZE']['std'])
                    mode.data = mode.data.float()

            # RGB transforms
            #
            # data[0].to_tensor()
            # if normalize:
            #     data[0].
            # for mode in data[1:]:
            #     if mode is not None:
            #         mode.to_tensor()
            #         mode.data = mode.data.float()
            #         print(type(mode), mode.data.size())

        return tuple([m.data for m in data if m is not None])

    def __getitem__(self, idx):
        # overwrite this function when creating a new dataset
        image = self.img_list[idx]
        mask_valid = np.ones(np.array(image).shape[:2])
        depth = None
        normals = None
        boundary = None

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)
        return sample


class PBRSDataset(GeoDataset):
    def __init__(self, img_list, root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_normals=True,
                 input_type='image'):
        super(PBRSDataset, self).__init__(img_list, root_dir=root_dir, img_size=img_size,
                                          transforms=transforms,
                                          use_boundary=use_boundary,
                                          use_depth=use_depth,
                                          use_normals=use_normals,
                                          input_type=input_type)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, 'img', img_name)
        image = Image.open(img_path)

        normals = None
        boundary = None
        depth = None

        mask_valid = imread(os.path.join(self.root_dir, 'normals', img_name.replace('_mlt.png', '_valid.png')))
        mask_valid = mask_valid / 255
        mask_valid = Mask(data=mask_valid.copy())
        image = InputImage(data=image)

        if self.use_depth:
            data = imread(os.path.join(self.root_dir, 'depth', img_name.replace('_mlt.png', '_depth.png')))
            data = data.astype('float32') / 65535.0
            depth = Depth(data=data.copy())

        if self.use_normals:
            data = imread(os.path.join(self.root_dir, 'normals', img_name.replace('_mlt.png', '_norm_camera.png')))
            normals_tmp = data.astype('float32')
            normals = Normals(data=normals_tmp.copy())
            normals.data[..., 0] = ((255 - normals_tmp[..., 0]) - 127.5) / 127.5
            normals.data[..., 1] = (normals_tmp[..., 2] - 127.5) / 127.5
            normals.data[..., 2] = -2.0 * ((255.0 - normals_tmp[..., 1]) / 255.0) + 1

        if self.use_boundary:
            data = imread(
                os.path.join(self.root_dir, 'boundaries', img_name.replace('_mlt.png', '_instance_boundary.png')))
            data = data / 255
            boundary = Contours(data=data.copy())

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)

        return sample


class NYUDataset(GeoDataset):
    def __init__(self, dataset_path, split_type='train', root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_normals=True,
                 input_type='image'):
        super(NYUDataset, self).__init__(img_list=None, root_dir=root_dir, img_size=img_size,
                                         transforms=transforms,
                                         use_boundary=use_boundary,
                                         use_depth=use_depth,
                                         use_normals=use_normals,
                                         input_type=input_type)
        self.dataset_path = os.path.join(root_dir, dataset_path)
        used_split = io.loadmat(os.path.join(root_dir, 'nyuv2_splits.mat'))
        self.idx_list = [idx[0] - 1 for idx in used_split[split_type + 'Ndxs']]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # Get image from NYUv2 mat file
        # Crop border by 6 pixels
        dataset = h5py.File(self.dataset_path, 'r', libver='latest', swmr=True)
        image = dataset['images'][self.idx_list[idx]]
        image_new = image.swapaxes(0, 2)

        normals = None
        boundary = None
        depth = None

        crop_ROI = [6, 6, 473, 630]
        image_new = image_new[crop_ROI[0]:crop_ROI[2], crop_ROI[1]:crop_ROI[3], :]

        mask_valid = np.ones(shape=image_new.shape[:2])
        mask_valid = Mask(data=mask_valid.copy())

        image_new = Image.fromarray(image_new)
        image = InputImage(data=image_new.copy())

        if self.use_depth:
            data = dataset['depths'][self.idx_list[idx]].swapaxes(0, 1).astype('float32') * 1000 / 65535
            data = data[crop_ROI[0]:crop_ROI[2], crop_ROI[1]:crop_ROI[3]]
            depth = Depth(data=data.copy())

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)
        return sample


class ClearGraspDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, normals_dir, boundary_dir, masks_dir, transform=None, input_only=None,
                 use_boundary=True,
                 use_depth=True,
                 use_normals=True,
                 normalize_input=True):
        # super(ClearGraspDataset, self).__init__(img_list=None,
        #                                  transforms=transforms,
        #                                  use_boundary=use_boundary,
        #                                  use_depth=use_depth,
        #                                  use_normals=use_normals)

        self.MAX_DEPTH = 3.0
        self.use_boundary = use_boundary
        self.use_depth = use_depth
        self.use_normals = use_normals
        self.transform = transform
        self.input_only = input_only

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.normal_dir = normals_dir
        self.boundary_dir = boundary_dir
        self.masks_dir = masks_dir
        self._datalist_rgb = []
        self._datalist_depth = []
        self._datalist_normal = []
        self._datalist_boundary = []
        self._datalist_mask = []
        self._extension_input = ['-transparent-rgb-img.jpg', '-rgb.jpg', '-input-img.jpg']
        self._extension_depth = ['depth-rectified.exr', 'transparent-depth-img.exr']
        self._extension_normal = ['-cameraNormals.exr', '-normals.exr']
        self._extension_boundary = ['outlineSegmentation.png',]
        self._extension_mask = ['-mask.png']
        self._create_lists_filenames(self.rgb_dir, self.depth_dir, self.normal_dir, self.boundary_dir, self.masks_dir)


    def __len__(self):
        return len(self._datalist_rgb)

    def __getitem__(self, idx):
        _img = imageio.imread(self._datalist_rgb[idx])

        if self.masks_dir:
            _mask = imageio.imread(self._datalist_mask[idx])

        normals = None
        boundary = None
        depth = None

        if self.use_depth:
            _depth = api_utils.exr_loader(self._datalist_depth[idx], ndim=1)
            _depth[np.isinf(_depth)] = 0
            _depth[np.isnan(_depth)] = 0
            _depth[_depth > self.MAX_DEPTH] = 0
            _depth = _depth * 1000 / 65535

        if self.use_normals:
            _normals_orig = api_utils.exr_loader(self._datalist_normal[idx])
            _normals_orig[np.isinf(_normals_orig)] = 0
            _normals_orig[np.isnan(_normals_orig)] = 0

            # Making all values of invalid pixels marked as -1.0 to 0.
            # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
            mask_invalid_pixels = np.all(_normals_orig == -1.0, axis=0)
            _normals_orig[:, mask_invalid_pixels] = 0.0

            # Convert normals into SharpNet format
            _normals = np.zeros_like(_normals_orig)
            _normals[0, ...] = -1 * _normals_orig[0, ...]
            _normals[1, ...] = _normals_orig[2, ...]
            _normals[2, ...] = -1 * _normals_orig[1, ...]

        if self.use_boundary:
            _boundary = imageio.imread(self._datalist_boundary[idx])
            _boundary[_boundary > 1] = 0  # Single channel PNG file with value of pixel denoting class
            _boundary = _boundary.astype(np.float32)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()

            _img = det_tf.augment_image(_img)
            if self.masks_dir:
                _mask = det_tf.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

            if self.use_normals:
                _normals = _normals.transpose((1, 2, 0))  # To Shape: (H, W, 3)
                _normals = det_tf.augment_image(_normals, hooks=ia.HooksImages(activator=self._activator_masks))
                _normals = _normals.transpose((2, 0, 1))  # To Shape: (3, H, W)

            if self.use_depth:
                _depth = det_tf.augment_image(_depth, hooks=ia.HooksImages(activator=self._activator_masks))

            if self.use_boundary:
                _boundary = det_tf.augment_image(_boundary, hooks=ia.HooksImages(activator=self._activator_masks))


        # Return Tensors
        _img_tensor = torchvision.transforms.ToTensor()(_img)
        _img_tensor = torchvision.transforms.Normalize(self.mean, self.std)(_img_tensor)
        if self.masks_dir:
            _mask = _mask[..., np.newaxis]
            _mask_tensor = torchvision.transforms.ToTensor()(_mask)
        else:
            _mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        if self.use_depth:
            _depth_tensor = torch.from_numpy(_depth)

        if self.use_normals:
            _normals_tensor = torch.from_numpy(_normals)
            _normals_tensor = nn.functional.normalize(_normals_tensor, p=2, dim=0)

        if self.use_boundary:
            _boundary_tensor = torch.from_numpy(_boundary)


        sample = [_img_tensor, _mask_tensor,
                  _depth_tensor if self.use_depth else None,
                  _normals_tensor if self.use_normals else None,
                  _boundary_tensor if self.use_boundary else None]
        sample = tuple([m for m in sample if m is not None])

        # TEST VALUES
        # for n, m in enumerate(sample):
        #     print('DataLoader Returns:', n, m.dtype, m.shape, m.min(), m.max())

        return sample

    def _create_lists_filenames(self, rgb_dir, depth_dir, normal_dir, boundary_dir, mask_dir):
        assert os.path.isdir(rgb_dir), 'Dataloader given images directory that does not exist: "%s"' % (rgb_dir)

        # make list of  normals files
        for ext in self._extension_input:
            self._datalist_rgb += sorted(glob.glob(os.path.join(rgb_dir, '*' + ext)))
        num_images = len(self._datalist_rgb)
        assert num_images > 0, 'No images found in given dir: {}'.format(rgb_dir)

        if depth_dir:
            assert os.path.isdir(depth_dir), ('Dataloader given labels directory that does not exist: "%s"'
                                               % (depth_dir))
            for ext in self._extension_depth:
                self._datalist_depth += sorted(glob.glob(os.path.join(depth_dir, '*' + ext)))
            assert len(self._datalist_depth) == num_images, 'Num of depth {} and rgb {} images not equal'.format(len(self._datalist_depth), num_images)

        if normal_dir:
            assert os.path.isdir(normal_dir), ('Dataloader given normals directory that does not exist: "%s"'
                                               % (normal_dir))
            for ext in self._extension_normal:
                self._datalist_normal += sorted(glob.glob(os.path.join(normal_dir, '*' + ext)))
            assert len(self._datalist_normal) == num_images, 'Num of normals {} and rgb {} images not equal'.format(len(self._datalist_normal), num_images)

        if boundary_dir:
            assert os.path.isdir(boundary_dir), ('Dataloader given outlines directory that does not exist: "%s"'
                                               % (boundary_dir))
            for ext in self._extension_boundary:
                self._datalist_boundary += sorted(glob.glob(os.path.join(boundary_dir, '*' + ext)))
            assert len(self._datalist_boundary) == num_images, 'Num of boundary {} and rgb {} images not equal'.format(len(self._datalist_boundary), num_images)

        if mask_dir:
            assert os.path.isdir(mask_dir), ('Dataloader given masks directory that does not exist: "%s"' %
                                               (mask_dir))
            for ext in self._extension_mask:
                self._datalist_mask += sorted(glob.glob(os.path.join(mask_dir, '*' + ext)))
            assert len(self._datalist_mask) == num_images, 'Num of mask {} and rgb {} images not equal'.format(len(self._datalist_mask), num_images)

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default