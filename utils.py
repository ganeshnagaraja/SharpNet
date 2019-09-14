from PIL import Image
import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_transforms import *
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision import transforms


def round_down(num, divisor):
    return num - (num % divisor)


def get_np_preds(image_pil, model, device, args):
    normals = None
    boundary = None
    depth = None

    image_np = np.array(image_pil)
    w, h = image_pil.size

    scale = args.rescale_factor

    h_new = round_down(int(h * scale), 16)
    w_new = round_down(int(w * scale), 16)

    if len(image_np.shape) == 2 or image_np.shape[-1] == 1:
        print("Input image has only 1 channel, please use an RGB or RGBA image")
        sys.exit(0)

    if len(image_np.shape) == 4 or image_np.shape[-1] == 4:
        # RGBA image to be converted to RGB
        image_pil = image_pil.convert('RGBA')
        image = Image.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
        image.paste(image_pil.copy(), mask=image_pil.split()[3])
    else:
        image = image_pil

    image = image.resize((w_new, h_new), Image.ANTIALIAS)

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = []
    t.extend([ToTensor(), normalize])
    transf = Compose(t)

    data = [image, None]
    image = transf(*data)

    image = torch.autograd.Variable(image).unsqueeze(0)
    image = image.to(device)

    if args.boundary:
        if args.depth and args.normals:
            depth_pred, normals_pred, boundary_pred = model(image)
            tmp = normals_pred.data.cpu()
        elif args.depth and not args.normals:
            depth_pred, boundary_pred = model(image)
            tmp = depth_pred.data.cpu()
        elif args.normals and not args.depth:
            normals_pred, boundary_pred = model(image)
            tmp = normals_pred.data.cpu()
        else:
            boundary_pred = model(image)
            tmp = boundary_pred.data.cpu()
    else:
        if args.depth:
            depth_pred = model(image)
            tmp = depth_pred.data.cpu()
        if args.depth and args.normals:
            depth_pred, normals_pred = model(image)
            tmp = normals_pred.data.cpu()
        if args.normals and not args.depth:
            normals_pred = model(image)
            tmp = normals_pred.data.cpu()

    shp = tmp.shape[2:]

    if args.normals:
        normals_pred = normals_pred.data.cpu().numpy()[0, ...]
        normals_pred = normals_pred.swapaxes(0, 1).swapaxes(1, 2)
        normals_pred[..., 0] = 0.5 * (normals_pred[..., 0] + 1)
        normals_pred[..., 1] = 0.5 * (normals_pred[..., 1] + 1)
        normals_pred[..., 2] = -0.5 * np.clip(normals_pred[..., 2], -1, 0) + 0.5

        normals_pred[..., 0] = normals_pred[..., 0] * 255
        normals_pred[..., 1] = normals_pred[..., 1] * 255
        normals_pred[..., 2] = normals_pred[..., 2] * 255

        normals = normals_pred.astype('uint8')

    if args.depth:
        depth_pred = depth_pred.data.cpu().numpy()[0, 0, ...] * 65535 / 1000
        depth_pred = (1 / scale) * cv2.resize(depth_pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        m = np.min(depth_pred)
        M = np.max(depth_pred)
        depth_pred = (depth_pred - m) / (M - m)
        depth = Image.fromarray(np.uint8(plt.cm.jet(depth_pred) * 255))
        depth = np.array(depth)[:, :, :3]

    if args.boundary:
        boundary_pred = boundary_pred.data.cpu().numpy()[0, 0, ...]
        boundary_pred = cv2.resize(boundary_pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        boundary_pred = np.clip(boundary_pred, 0, 10)
        boundary = (boundary_pred * 255).astype('uint8')

    return tuple([depth, normals, boundary])


def get_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                yield p


def freeze_model_decoders(model, freeze_decoders):
    if 'normals' in freeze_decoders:
        model.normals_decoder.freeze()
    if 'depth' in freeze_decoders:
        model.depth_decoder.freeze()
    if 'boundary' in freeze_decoders:
        model.boundary_decoder.freeze()


def get_gt_sample(dataloader, loader_iter, args):
    try:
        data = next(loader_iter)
    except:
        loader_iter = iter(dataloader)
        data = next(loader_iter)

    if args.depth:
        if args.boundary and args.normals:
            if len(data) == 5:
                # normals and boundary GT
                input, mask_gt, depth_gt, normals_gt, boundary_gt = data
            else:
                # NYU
                input, mask_gt, depth_gt = data
                normals_gt = None
                boundary_gt = None
        elif args.boundary and not args.normals:
            input, mask_gt, depth_gt, boundary_gt = data
        elif args.boundary:
            input, mask_gt, depth_gt, normals_gt = data
        else:
            input, mask_gt, depth_gt = data
    else:
        depth_gt = None
        boundary_gt = None
        if args.boundary and args.normals:
            input, mask_gt, normals_gt, boundary_gt = data
        elif args.normals and not args.boundary:
            input, mask_gt, normals_gt = data

    input = input.cuda(async=False)
    mask_gt = mask_gt.cuda(async=False)
    if normals_gt is not None:
        normals_gt = normals_gt.cuda(async=False)
        normals_gt = torch.autograd.Variable(normals_gt)
    if depth_gt is not None:
        depth_gt = depth_gt.cuda(async=False)
        depth_gt = torch.autograd.Variable(depth_gt)
    if boundary_gt is not None:
        boundary_gt = boundary_gt.cuda(async=False)
        boundary_gt = torch.autograd.Variable(boundary_gt)

    input = torch.autograd.Variable(input)
    mask_gt = torch.autograd.Variable(mask_gt)
    return input, mask_gt, depth_gt, normals_gt, boundary_gt


def write_loss_components(tb_writer, iteration, epoch, dataset_size, args,
                          depth_loss_meter=None, depth_loss=None,
                          normals_loss_meter=None, normals_loss=None,
                          boundary_loss_meter=None, boundary_loss=None,
                          grad_loss_meter=None, grad_loss=None,
                          consensus_loss_meter=None, consensus_loss=None):

    if args.normals and normals_loss_meter is not None:
        if args.verbose:
            print('Normals loss: ' + str(float(normals_loss)))
        tb_writer.add_scalar("normals_loss", normals_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
    if args.depth and depth_loss_meter is not None:
        if args.verbose:
            print('Depth loss: ' + str(float(depth_loss)))
            print('Gradient loss: ' + str(float(grad_loss)))
        tb_writer.add_scalar("Depth_loss", depth_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
        tb_writer.add_scalar("grad_loss", grad_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
    if args.boundary and boundary_loss_meter is not None:
        if args.verbose:
            print('Boundary loss: ' + str(float(boundary_loss)))
        tb_writer.add_scalar("boundary loss", boundary_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
    if args.geo_consensus and consensus_loss_meter is not None:
        if args.verbose:
            print('Consensus loss: ' + str(float(consensus_loss)))
        tb_writer.add_scalar("consensus loss", consensus_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)


def get_tensor_preds(input, model, args):
    depth_pred = None
    normals_pred = None
    boundary_pred = None
    if args.depth:
        if args.boundary and args.normals:
            depth_pred, normals_pred, boundary_pred = model(input)
        elif args.boundary and not args.normals:
            depth_pred, boundary_pred = model(input)
        elif args.normals:
            depth_pred, normals_pred = model(input)
        else:
            depth_pred = model(input)
    else:
        if args.boundary and args.normals:
            normals_pred, boundary_pred = model(input)
        elif args.boundary and not args.normals:
            boundary_pred = model(input)
        else:
            normals_pred = model(input)

    return depth_pred, normals_pred, boundary_pred


def adjust_learning_rate(lr, lr_mode, step, max_epoch, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if lr_mode == 'step':
        lr = lr * (0.1 ** (epoch // step))
    elif lr_mode == 'poly':
        lr = lr * (1 - epoch / max_epoch) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    # camera_normal_rgb *= 127.5
    # camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb

def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0, max_depth=1.0):
    '''Converts a floating point depth image to uint8 or uint16 image.
    The depth image is first scaled to (0.0, max_depth) and then scaled and converted to given datatype.

    Args:
        depth_img (numpy.float32): Depth image, value is depth in meters
        dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type. Must be np.uint8 or np.uint16
        max_depth (float, optional): The max depth to be considered in the input depth image. The min depth is
            considered to be 0.0.
    Raises:
        ValueError: If wrong dtype is given

    Returns:
        numpy.ndarray: Depth image scaled to given dtype
    '''

    if dtype != np.uint16 and dtype != np.uint8:
        raise ValueError('Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'.format(dtype))

    # Clip depth image to given range
    depth_img = np.ma.masked_array(depth_img, mask=(depth_img == 0.0))
    depth_img = np.ma.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    min_val = type_info.min
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    depth_img = np.ma.filled(depth_img, fill_value=0)  # Convert back to normal numpy array from masked numpy array

    return depth_img

def depth2rgb(depth_img, min_depth=0.0, max_depth=1.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
              dynamic_scaling=False):
    '''Generates RGB representation of a depth image.
    To do so, the depth image has to be normalized by specifying a min and max depth to be considered.

    Holes in the depth image (0.0) appear black in color.

    Args:
        depth_img (numpy.ndarray): Depth image, values in meters. Shape=(H, W), dtype=np.float32
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
                          Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html

                          Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
                          This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
                                image, rather that the passed arguments.
    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
    '''
    # Map depth image to Color Map
    if dynamic_scaling:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8,
                                                min_depth=max(depth_img[depth_img > 0].min(), min_depth),    # Add a small epsilon so that min depth does not show up as black (invalid pixels)
                                                max_depth=min(depth_img.max(), max_depth))
    else:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

    if reverse_scale is True:
        depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
        depth_img_scaled = 255 - depth_img_scaled
        depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

    depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[depth_img_scaled == 0, :] = 0

    return depth_img_mapped


def create_grid_image(inputs, normals_gt, normals_pred, depth_gt, depth_pred, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''
    min_depth = 0.05
    max_depth = 10.0
    depth_gt = depth_gt * 65535 / 1000 # de-normalizing depth
    depth_pred = depth_pred * 65535 / 1000  # de-normalizing depth

    # inverse normalize rgb image
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    inputs[:, 0, :, :] = inputs[:, 0, :, :] * std[0] + mean[0]
    inputs[:, 1, :, :] = inputs[:, 1, :, :] * std[1] + mean[1]
    inputs[:, 2, :, :] = inputs[:, 2, :, :] * std[2] + mean[2]

    # normals_pred[normals_pred < 0] = 0

    normals_gt[:, 0, :, :] = normals_gt[:, 0, :, :] * -1
    normals_gt[:, 1, :, :] = normals_gt[:, 1, :, :]
    normals_gt[:, 2, :, :] = normals_gt[:, 2, :, :] * -1

    normals_pred[:, 0, :, :] = normals_pred[:, 0, :, :] * -1
    normals_pred[:, 1, :, :] = normals_pred[:, 1, :, :]
    normals_pred[:, 2, :, :] = normals_pred[:, 2, :, :] * -1

    img_tensor = inputs[:max_num_images_to_save]
    img_tensor_normals_gt = normals_gt[:max_num_images_to_save]
    img_tensor_normals_pred = normals_pred[:max_num_images_to_save]
    img_tensor_depth_gt = depth_gt[:max_num_images_to_save]
    img_tensor_depth_pred = depth_pred[:max_num_images_to_save]


    images = []
    for h, i, j, k, l in zip(img_tensor, img_tensor_normals_gt, img_tensor_normals_pred, img_tensor_depth_gt, img_tensor_depth_pred):
        i = transforms.ToTensor()(normal_to_rgb(i.numpy().transpose(1, 2, 0)))  # Normals to RGB
        j = transforms.ToTensor()(normal_to_rgb(j.numpy().transpose(1, 2, 0)))
        k = transforms.ToTensor()(depth2rgb(k.squeeze(0).numpy(),
                                                      min_depth=min_depth,
                                                      max_depth=max_depth))
        l_stat = transforms.ToTensor()(depth2rgb(l.squeeze(0).numpy(),
                                                      min_depth=min_depth,
                                                      max_depth=max_depth))
        l_dyn = transforms.ToTensor()(depth2rgb(l.squeeze(0).numpy(),
                                                      min_depth=min_depth,
                                                      max_depth=max_depth,
                                                      dynamic_scaling=True))

        images.extend([h, i, j, k, l_stat, l_dyn])
    grid_image = make_grid(images, nrow=6, normalize=False, scale_each=False)

    return grid_image
