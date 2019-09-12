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


class Synthetic(GeoDataset):
    def __init__(self, rgb_dir, depth_dir, normals_dir, boundary_dir, split_type='train', root_dir='', img_size=480, transforms=None,
                 use_boundary=False,
                 use_depth=True,
                 use_normals=True,
                 input_type='image'):
        super(Synthetic, self).__init__(img_list=None, root_dir=root_dir, img_size=img_size,
                                         transforms=transforms,
                                         use_boundary=use_boundary,
                                         use_depth=use_depth,
                                         use_normals=use_normals,
                                         input_type=input_type)

        self.MAX_DEPTH = 3.0
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.normal_dir = normals_dir
        self.boundary_dir = boundary_dir
        self._datalist_rgb = []
        self._datalist_depth = []
        self._datalist_normal = []
        self._datalist_boundary = []
        self._extension_input = ['.jpg']
        self._extension_depth = ['.exr']
        self._extension_normal = ['.exr']
        self._extension_boundary = ['.png']
        self._create_lists_filenames(self.rgb_dir, self.depth_dir, self.normal_dir, self.boundary_dir)


    def __len__(self):
        return len(self._datalist_rgb)

    def __getitem__(self, idx):
        image = imageio.imread(self._datalist_rgb[idx])
        image_new = image


        normals = None
        boundary = None
        depth = None

        mask_valid = np.ones(shape=image_new.shape[:2], dtype=np.uint8)
        mask_valid = Mask(data=mask_valid.copy())

        image_new = Image.fromarray(image_new)
        image = InputImage(data=image_new.copy())

        if self.use_depth:
            data = api_utils.exr_loader(self._datalist_depth[idx], ndim=1)
            data[np.isinf(data)] = 0
            data[np.isnan(data)] = 0
            data[data > self.MAX_DEPTH] = 0
            data = data * 1000 / 65535

            depth = Depth(data=data.copy())

        if self.use_normals:
            data = api_utils.exr_loader(self._datalist_normal[idx])
            data = data.transpose(1, 2, 0).astype(np.float64)
            data[np.isinf(data)] = 0
            data[np.isnan(data)] = 0

            normals = Normals(data=data.copy())
            normals.data[...,0] = normals.data[...,0] * -1
            normals.data[...,1] = data[...,2]
            normals.data[...,2] = normals.data[...,1] * -1


        if self.use_boundary:
            data = imageio.imread(self._datalist_boundary[idx])
            data[data > 1] = 0  # Single channel PNG file with value of pixel denoting class
            data = data.astype(np.float32)

            boundary = Contours(data=data.copy())

        sample = self.format_data(image,
                                  mask_valid=mask_valid,
                                  depth=depth,
                                  normals=normals,
                                  boundary=boundary)
        return sample

    def _create_lists_filenames(self, rgb_dir, depth_dir, normal_dir, boundary_dir):
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
            assert len(self._datalist_depth) == num_images, 'Num of depth and rgb images not equal'

        if normal_dir is not None:
            assert os.path.isdir(normal_dir), ('Dataloader given normals directory that does not exist: "%s"'
                                               % (normal_dir))
            for ext in self._extension_normal:
                self._datalist_normal += sorted(glob.glob(os.path.join(normal_dir, '*' + ext)))
            assert len(self._datalist_normal) == num_images, 'Num of depth and rgb images not equal'

        if boundary_dir is not None:
            assert os.path.isdir(normal_dir), ('Dataloader given outlines directory that does not exist: "%s"'
                                               % (boundary_dir))
            for ext in self._extension_boundary:
                self._datalist_boundary += sorted(glob.glob(os.path.join(boundary_dir, '*' + ext)))
            assert len(self._datalist_boundary) == num_images, 'Num of depth and rgb images not equal'