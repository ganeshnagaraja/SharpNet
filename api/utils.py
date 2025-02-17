'''Functions for reading and saving EXR images using OpenEXR.
'''
import struct

# import open3d as o3d
import numpy as np
import cv2
import Imath
import OpenEXR
from PIL import Image
import torch


# import torch
# from torchvision.utils import make_grid


def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def exr_saver(EXR_PATH, ndarr, ndim=3):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array containing img data
        ndim (int): The num of dimensions in the saved exr image, either 3 or 1.
                        If ndim = 3, ndarr should be of shape (height, width) or (3 x height x width),
                        If ndim = 1, ndarr should be of shape (height, width)
    Returns:
        None
    '''
    if ndim == 3:
        # Check params
        if len(ndarr.shape) == 2:
            # If a depth image of shape (height x width) is passed, convert into shape (3 x height x width)
            ndarr = np.stack((ndarr, ndarr, ndarr), axis=0)

        if ndarr.shape[0] != 3 or len(ndarr.shape) != 3:
            raise ValueError(
                'The shape of the tensor should be (3 x height x width) for ndim = 3. Given shape is {}'.format(
                    ndarr.shape))

        # Convert each channel to strings
        Rs = ndarr[0, :, :].astype(np.float16).tostring()
        Gs = ndarr[1, :, :].astype(np.float16).tostring()
        Bs = ndarr[2, :, :].astype(np.float16).tostring()

        # Write the three color channels to the output file
        HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})
        out.close()
    elif ndim == 1:
        # Check params
        if len(ndarr.shape) != 2:
            raise ValueError(('The shape of the tensor should be (height x width) for ndim = 1. ' +
                              'Given shape is {}'.format(ndarr.shape)))

        # Convert each channel to strings
        Rs = ndarr[:, :].astype(np.float16).tostring()

        # Write the color channel to the output file
        HEADER = OpenEXR.Header(ndarr.shape[1], ndarr.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "R"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs})
        out.close()


def save_uint16_png(path, image):
    '''save weight file - scaled png representation of outlines estimation

        Args:
            path (str): path to save the file
            image (numpy.ndarray): 16-bit single channel image to be saved.
                                          Shape=(H, W), dtype=np.uint16
        '''
    assert image.dtype == np.uint16, ("data type of the array should be np.uint16." + "Got {}".format(image.dtype))
    assert len(image.shape) == 2, ("Shape of input image should be (H, W)" + "Got {}".format(len(image.shape)))

    array_buffer = image.tobytes()
    img = Image.new("I", image.T.shape)
    img.frombytes(array_buffer, 'raw', 'I;16')
    img.save(path)


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
    depth_img = np.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    min_val = type_info.min
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    return depth_img


def depth2rgb(depth_img, min_depth=0.0, max_depth=1.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False):
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

    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
    '''
    # Map depth image to Color Map
    depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)
    # depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=depth_img[depth_img>0].min(), max_depth=min(depth_img.max(), max_depth))

    if reverse_scale is True:
        depth_img_scaled = 255 - depth_img_scaled

    depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[depth_img_scaled == 0, :] = 0

    return depth_img_mapped


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
    camera_normal_rgb = (normals_to_convert + 1)
    camera_normal_rgb *= 127.5
    camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb


def scale_depth(depth_image):
    '''Convert depth in meters (float32) to a scaled uint16 format as required by depth2depth module.

    Args:
        depth_image (numpy.ndarray, float32): Depth Image

    Returns:
        numpy.ndarray: scaled depth image. dtype=np.uint16
    '''

    assert depth_image.dtype == np.float32, "data type of the array should be float32. Got {}".format(depth_image.dtype)
    SCALING_FACTOR = 4000
    OUTPUT_DTYPE = np.uint16

    # Prevent Overflow of data by clipping depth values
    type_info = np.iinfo(OUTPUT_DTYPE)
    max_val = type_info.max
    depth_image = np.clip(depth_image, 0, np.floor(max_val / SCALING_FACTOR))

    return (depth_image * SCALING_FACTOR).astype(OUTPUT_DTYPE)


def unscale_depth(depth_image):
    '''Unscale the depth image from uint16 to denote the depth in meters (float32)

    Args:
        depth_image (numpy.ndarray, uint16): Depth Image

    Returns:
        numpy.ndarray: unscaled depth image. dtype=np.float32
    '''

    assert depth_image.dtype == np.uint16, "data type of the array should be uint16. Got {}".format(depth_image.dtype)
    SCALING_FACTOR = 4000

    return depth_image.astype(np.float32) / SCALING_FACTOR


def normal_to_rgb(normals_to_convert, output_dtype='float'):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
        output_dtype (str): format of output, possibel values = ['float', 'uint8']
                            if 'float', range of output (0,1)
                            if 'uint8', range of output (0,255)
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    if output_dtype == 'uint8':
        camera_normal_rgb *= 255
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    elif output_dtype == 'float':
        pass
    else:
        raise NotImplementedError('possibel values are only float and uint8. received value {}'.format(output_dtype))
    return camera_normal_rgb


# def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
#     '''Make a grid of images for display purposes
#     Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

#     Args:
#         inputs (Tensor): Batch Tensor of shape (B x C x H x W)
#         outputs (Tensor): Batch Tensor of shape (B x C x H x W)
#         labels (Tensor): Batch Tensor of shape (B x C x H x W)
#         max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
#             max number of imaged to put in grid

#     Returns:
#         numpy.ndarray: A numpy array with of input images arranged in a grid
#     '''

#     img_tensor = inputs[:max_num_images_to_save]

#     output_tensor = outputs[:max_num_images_to_save]
#     output_tensor_rgb = normal_to_rgb(output_tensor)

#     label_tensor = labels[:max_num_images_to_save]
#     label_tensor_rgb = normal_to_rgb(label_tensor)

#     images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb), dim=3)
#     grid_image = make_grid(images, 1, normalize=True, scale_each=True)

#     return grid_image


def _get_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    """Creates point cloud from rgb images and depth image

    Args:
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len along x-axis in pixels of camera used to capture image.
        fy (int): The focal len along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    Returns:
        numpy.ndarray: camera_points - The XYZ location of each pixel. Shape: (num of pixels, 3)
        numpy.ndarray: color_points - The RGB color of each pixel. Shape: (num of pixels, 3)
    """
    # camera instrinsic parameters
    # camera_intrinsics  = [[fx 0  cx],
    #                       [0  fy cy],
    #                       [0  0  1]]
    camera_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                                   np.linspace(0, image_height - 1, image_height))
    camera_points_x = np.multiply(pixel_x - camera_intrinsics[0, 2], (depth_image / camera_intrinsics[0, 0]))
    camera_points_y = np.multiply(pixel_y - camera_intrinsics[1, 2], (depth_image / camera_intrinsics[1, 1]))
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x, camera_points_y, camera_points_z]).transpose(1, 2, 0).reshape(-1, 3)

    color_points = color_image.reshape(-1, 3)

    # Do not Remove invalid 3D points (where depth == 0), since it results in unstructured point cloud, which is not easy to work with using Open3D
    # valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    # camera_points = camera_points[valid_depth_ind, :]
    # color_points = color_points[valid_depth_ind, :]

    return camera_points, color_points


def write_point_cloud(filename, color_image, depth_image, fx, fy, cx, cy):
    """Creates and Writes a .ply point cloud file using RGB and Depth images.

    Args:
        filename (str): The path to the file which should be written. It should end with extension '.ply'
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len along x-axis in pixels of camera used to capture image.
        fy (int): The focal len along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    """
    xyz_points, rgb_points = _get_point_cloud(color_image, depth_image, fx, fy, cx, cy)

    # Open3D Normals Estimation
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz_points)
    # # Estimate Normals
    # o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
    # o3d.geometry.orient_normals_towards_camera_location(pcd, camera_location=np.array([0., 0., 0.]))
    # pcd.normalize_normals()
    # pcd_normals = np.asarray(pcd.normals)
    # pcd_normals = np.reshape(pcd_normals, (288, 512, 3))
    # pcd_normals[:, :, 1] *= - 1
    # pcd_normals[:, :, 2] *= - 1

    # pcd_normals_rgb = normal_to_rgb(pcd_normals, output_dtype='uint8')
    # rgb_points = pcd_normals_rgb.reshape(-1, 3)

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(
                bytearray(
                    struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(), rgb_points[i, 2].tostring())))


def compute_errors(gt, pred, mask):

    """Compute error for depth as required for paper (RMSE, REL, etc)
        Args:
            gt (numpy.ndarray): Ground truth depth (metric). Shape: [B, H, W], dtype: float32
            pred (numpy.ndarray): Predicted depth (metric). Shape: [B, H, W], dtype: float32
            mask (numpy.ndarray): Mask of pixels to consider while calculating error.
                                  Pixels not in mask are ignored and do not contribute to error.
                                  Shape: [B, H, W], dtype: bool

        Returns:
            dict: Various measures of error metrics
        """

    safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))

    # mask_valid_region = mask
    # depth_gt = gt

    gt = torch.from_numpy(gt)
    pred = torch.from_numpy(pred)
    mask = torch.from_numpy(mask.astype(np.uint8)).byte()

    gt = gt[mask]
    pred = pred[mask]
    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.05).float().mean()
    a2 = (thresh < 1.10).float().mean()
    a3 = (thresh < 1.25).float().mean()

    rmse = ((gt - pred)**2).mean().sqrt()
    rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()
    log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
    abs_rel = ((gt - pred).abs() / gt).mean()
    mae = (gt - pred).abs().mean()
    sq_rel = ((gt - pred)**2 / gt).mean()

    measures = {
        'a1': round(a1.item() * 100, 5),
        'a2': round(a2.item() * 100, 5),
        'a3': round(a3.item() * 100, 5),
        'rmse': round(rmse.item(), 5),
        'rmse_log': round(rmse_log.item(), 5),
        'log10': round(log10.item(), 5),
        'abs_rel': round(abs_rel.item(), 5),
        'sq_rel': round(sq_rel.item(), 5),
        'mae': round(mae.item(), 5),
    }
    return measures
    # """Compute error for depth as required for paper (RMSE, REL, etc)
    # Args:
    #     gt (torch.Tensor): Ground truth depth (metric) shape [batch, h, w]
    #     pred (torch.Tensor): Predicted depth. shape [batch, h, w]
    #     mask (torch.Tensor): Mask of pixels to consider while calculating error.
    #                             Pixels not in mask are ignored and do not contribute to error.
    #                             shape [batch, h, w]
    #                             dtype = bool

    # Returns:
    #     dict: Various measures of error metrics
    # """
    # print('...................................')
    # print('label max, min ', gt.max(), gt.min())
    # print('output max, min ',pred.max(), pred.min())
    # print('.................................')
    # # mask = mask > 0
    # safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    # safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    # # batch_size = pred.shape[0]
    # mask_holes = gt > 0
    # mask = (mask_holes) & (mask)
    # gt = gt[mask]
    # pred = pred[mask]
    # thresh = torch.max(gt / pred, pred / gt)
    # a1 = (thresh < 1.25).float().mean()  # * batch_size
    # a2 = (thresh < 1.25**2).float().mean()  # * batch_size
    # a3 = (thresh < 1.25**3).float().mean()  # * batch_size

    # rmse = ((gt - pred)**2).mean().sqrt()  # * batch_size
    # rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()  # * batch_size
    # log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()  # * batch_size
    # abs_rel = ((gt - pred).abs() / gt).mean()  # * batch_size
    # mae = (gt - pred).abs().mean()
    # sq_rel = ((gt - pred)**2 / gt).mean()  # * batch_size
    # measures = {
    #     'a1': a1,
    #     'a2': a2,
    #     'a3': a3,
    #     'rmse': rmse,
    #     'rmse_log': rmse_log,
    #     'log10': log10,
    #     'abs_rel': abs_rel,
    #     'sq_rel': sq_rel,
    #     'mae': mae
    # }
    # return measures