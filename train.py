import torch
import datetime
from torch.optim import SGD, Adam
import argparse
import oyaml
from attrdict import AttrDict
from termcolor import colored

from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import *
from dataset_manager import *
from sharpnet_model import *
from loss import *
from resnet import Bottleneck as ResBlock
from utils import *
import torch.nn as nn

import os
import sys


def train_epoch(train_loader, val_loader, model, criterion, optimizer, epoch,
                train_writer, val_writer,
                train_loss_meter, val_loss_meter,
                depth_loss_meter, grad_loss_meter,
                normals_loss_meter,
                date_str, model_save_path,
                config,
                boundary_loss_meter=None, consensus_loss_meter=None):

    batch_size = int(config.train.batch_size)
    iter_size = config.train.iter_size
    num_workers = int(config.train.num_workers)

    loader_iter = iter(train_loader)

    for iter_i, _ in enumerate(train_loader):
        optimizer.zero_grad()
        iter_loss = 0
        iter_normals_loss = 0
        iter_grad_loss = 0
        iter_depth_loss = 0
        iter_boundary_loss = 0
        iter_consensus_loss = 0

        freeze_decoders = config.train.decoder_freeze.split(',')
        freeze_model_decoders(model, freeze_decoders)

        # accumulated gradients
        for i in range(iter_size):
            # get ground truth sample
            input, mask_gt, depth_gt, normals_gt, boundary_gt = get_gt_sample(train_loader, loader_iter, config.train)

            # compute output
            depth_pred, normals_pred, boundary_pred = get_tensor_preds(input, model, config.train)

            # compute loss
            depth_loss, grad_loss, normals_loss, b_loss, geo_loss = criterion(mask_gt,
                                                                              d_pred=depth_pred,
                                                                              d_gt=depth_gt,
                                                                              n_pred=normals_pred,
                                                                              n_gt=normals_gt,
                                                                              b_pred=boundary_pred,
                                                                              b_gt=boundary_gt,
                                                                              use_grad=True)

            loss_real = depth_loss + grad_loss + normals_loss + b_loss + geo_loss
            loss = 1 * depth_loss + 0.1 * grad_loss + 0.5 * normals_loss + 0.005 * b_loss + 0.5 * geo_loss
            loss_real /= float(iter_size)
            loss /= float(iter_size)

            iter_loss += float(loss_real)
            iter_normals_loss += float(normals_loss)

            if grad_loss != 0:
                iter_grad_loss += float(grad_loss)
            if depth_loss != 0:
                iter_depth_loss += float(depth_loss)
            if b_loss != 0:
                iter_boundary_loss += float(b_loss)
            if geo_loss != 0:
                iter_consensus_loss += float(geo_loss)

            loss.backward()

        parameters = get_params(model)
        clip_grad_norm_(parameters, 10.0, norm_type=2)
        optimizer.step()

        if iter_normals_loss != 0:
            iter_normals_loss /= float(iter_size)
            normals_loss_meter.add(float(normals_loss))
        if iter_depth_loss != 0:
            iter_depth_loss /= float(iter_size)
            depth_loss_meter.add(float(iter_depth_loss))
        if iter_grad_loss != 0:
            iter_grad_loss /= float(iter_size)
            grad_loss_meter.add(float(iter_grad_loss))
        if iter_boundary_loss != 0:
            iter_boundary_loss /= float(iter_size)
            boundary_loss_meter.add(float(iter_boundary_loss))
        if iter_consensus_loss != 0:
            iter_consensus_loss /= float(iter_size)
            consensus_loss_meter.add(float(iter_consensus_loss))

        train_size = len(train_loader.dataset)
        iter_per_epoch = int(train_size/config.train.batch_size)
        train_loss_meter.add(float(iter_loss))
        print("\nepoch: " + str(epoch) + " | iter: {}/{} ".format(iter_i, iter_per_epoch) + "| Train Loss: " + str(float(iter_loss)) + '\n')
        train_writer.add_scalar("train_loss", train_loss_meter.value()[0],
                                int(epoch) * iter_per_epoch + iter_i)

        write_loss_components(train_writer, iter_i, epoch, train_size, config.train,
                              depth_loss_meter, iter_depth_loss,
                              normals_loss_meter, iter_normals_loss,
                              boundary_loss_meter, iter_boundary_loss,
                              grad_loss_meter, iter_grad_loss,
                              consensus_loss_meter, iter_consensus_loss)

        normals_gt = normals_gt.detach().cpu() if normals_gt is not None else torch.ones_like(input, dtype=torch.float32)
        normals_pred = normals_pred.detach().cpu() if normals_gt is not None else torch.ones_like(input, dtype=torch.float32)

        grid_image = create_grid_image(input.detach().cpu(),
                                             normals_gt.detach().cpu() , normals_pred.detach().cpu().float(),
                                             depth_gt.detach().cpu(), depth_pred.detach().cpu(),
                                             max_num_images_to_save=16)
        train_writer.add_image('Train image', grid_image, int(epoch) * iter_per_epoch + iter_i)

        if (iter_i + 1) % 50 == 0:
            val_loss = 0
            val_depth_loss = 0
            val_grad_loss = 0
            val_normals_loss = 0
            val_boundary_loss = 0
            val_consensus_loss = 0

            val_size = len(val_loader.dataset)

            with torch.no_grad():
                # evaluate on validation set
                model.eval()
                loader_iter = iter(val_loader)

                n_val_batches = int(float(val_size) / batch_size)
                for i in range(n_val_batches)[:50]:
                    # get ground truth sample
                    input, mask_gt, depth_gt, normals_gt, boundary_gt = get_gt_sample(val_loader, loader_iter, config.train)
                    # compute output
                    depth_pred, normals_pred, boundary_pred = get_tensor_preds(input, model, config.train)
                    # compute loss
                    depth_loss, grad_loss, normals_loss, b_loss, geo_loss = criterion(mask_gt,
                                                                                      d_pred=depth_pred,
                                                                                      d_gt=depth_gt,
                                                                                      n_pred=normals_pred,
                                                                                      n_gt=normals_gt,
                                                                                      b_pred=boundary_pred,
                                                                                      b_gt=boundary_gt,
                                                                                      use_grad=True)

                    iter_loss = depth_loss + normals_loss + grad_loss + b_loss + geo_loss

                    iter_loss = float(iter_loss) / 50
                    val_loss += iter_loss
                    if grad_loss != 0:
                        val_grad_loss += float(grad_loss) / 50
                    if depth_loss != 0:
                        val_depth_loss += float(depth_loss) / 50
                    if b_loss != 0:
                        val_boundary_loss += float(b_loss) / 50
                    if geo_loss != 0:
                        val_consensus_loss += float(geo_loss) / 50
                    if normals_loss != 0:
                        val_normals_loss += float(normals_loss) / 50

            val_loss_meter.add(val_loss)
            print("epoch: " + str(epoch) + " | iter: {}/{} ".format(iter_i, iter_per_epoch) + "| Val Loss: " + str(
                float(val_loss)))
            val_writer.add_scalar("val_loss", val_loss_meter.value()[0],
                                  int(epoch) * iter_per_epoch + iter_i)

            write_loss_components(val_writer, iter_i, epoch, train_size, config.train,
                                  depth_loss_meter, val_depth_loss,
                                  normals_loss_meter, val_normals_loss,
                                  boundary_loss_meter, val_boundary_loss,
                                  grad_loss_meter, val_grad_loss,
                                  consensus_loss_meter, val_consensus_loss)
            
            grid_image = create_grid_image(input.detach().cpu(),
                                             normals_gt.detach().cpu(),normals_pred.detach().cpu().float(),
                                             depth_gt.detach().cpu(), depth_pred.detach().cpu(),
                                             max_num_images_to_save=16)
            val_writer.add_image('Val image', grid_image, int(epoch) * iter_per_epoch + iter_i)

            model.train()

            freeze_decoders = config.train.decoder_freeze.split(',')
            freeze_model_decoders(model, freeze_decoders)

        if (iter_i + 1) % 1000 == 0:
            print('Saving checkpoint')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, "checkpoint_{}_iter_{}.pth".format(epoch, iter_i + 1)),
            )
            print('Done')


def get_trainval_splits(config):

    args = config.train
    t = {'NORMALIZE': {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
         'RESIZE_TRANSPARENT': 256
         }

    # t = {'SCALE': 2,
    #      'CROP': 320,
    #      'HORIZONTALFLIP': 1,
    #      'ROTATE': 6,
    #      'GAMMA': 0.15,
    #      'NORMALIZE': {'mean': [0.485, 0.456, 0.406],
    #                    'std': [0.229, 0.224, 0.225]}
    #      }
    # if args.dataset != 'NYU':
    #     try:
    #         with open(os.path.join(args.root_dir, 'jobs_train.txt'), 'r') as f:
    #             list_train_files = [line.split('\n')[0] for line in f.readlines() if line != '\n']
    #     except Exception as e:
    #         print('The file containing the list of images does not exist')
    #         print(os.path.join(args.root_dir, 'jobs_train.txt'))
    #         sys.exit(0)

    #     try:
    #         with open(os.path.join(args.root_dir, 'jobs_val.txt'), 'r') as f:
    #             list_val_files = [line.split('\n')[0] for line in f.readlines() if line != '\n']
    #     except Exception as e:
    #         print('The file containing the list of images does not exist')
    #         print(os.path.join(args.root_dir, 'jobs_val.txt'))
    #         sys.exit(0)

    #     if len(list_train_files) < 2:
    #         print('Train file contains less than 2 files, error')
    #         sys.exit(0)
    #     if len(list_val_files) < 2:
    #         print('Val file contains less than 2 files, error')
    #         sys.exit(0)

    #     train_files = list_train_files
    #     val_files = list_val_files

    if args.dataset == 'PBRS':
        train_dataset = PBRSDataset(img_list=train_files, root_dir=args.root_dir,
                                    transforms=t,
                                    use_depth=True if args.depth else False,
                                    use_boundary=True if args.boundary else False,
                                    use_normals=True if args.normals else False)
        val_dataset = PBRSDataset(img_list=val_files, root_dir=args.root_dir,
                                  transforms=t,
                                  use_depth=True if args.depth else False,
                                  use_boundary=True if args.boundary else False,
                                  use_normals=True if args.normals else False)
    elif args.dataset == 'NYU':
        train_dataset = NYUDataset('nyu_depth_v2_labeled.mat', split_type='train', root_dir=args.root_dir,
                                   transforms=t,
                                   use_depth=True,
                                   use_boundary=False,
                                   use_normals=False)
        val_dataset = NYUDataset('nyu_depth_v2_labeled.mat', split_type='test', root_dir=args.root_dir,
                                 transforms=t,
                                 use_depth=True,
                                 use_boundary=False,
                                 use_normals=False)
    elif args.dataset == 'clear_grasp':
        train_loader_list = []
        for dataset in config.train.datasetsTrain:
            train_dataset = Synthetic(dataset.rgb, dataset.depth, dataset.normals, dataset.outlines, split_type='train', root_dir=args.root_dir,
                                   transforms=t,
                                   use_depth=True,
                                   use_boundary=True,
                                   use_normals=True)
            train_loader_list.append(train_dataset)

        test_loader_list = []
        for dataset in config.train.datasetsVal:
            val_dataset = Synthetic(dataset.rgb, dataset.depth, dataset.normals, dataset.outlines, split_type='test', root_dir=args.root_dir,
                                 transforms=t,
                                 use_depth=True,
                                 use_boundary=True,
                                 use_normals=True)
            test_loader_list.append(val_dataset)

        train_dataset = torch.utils.data.ConcatDataset(train_loader_list)
        val_dataset = torch.utils.data.ConcatDataset(test_loader_list)

    train_dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=int(args.num_workers))

    val_dataloader = DataLoader(val_dataset, batch_size=int(args.batch_size),
                                shuffle=True, num_workers=int(args.num_workers))

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train the SharpNet network")
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    args = parser.parse_args()

    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = oyaml.load(fd)  # Returns an ordered dict. Used for printing

    config = AttrDict(config_yaml)
    print(colored('Config being used for training:\n{}\n\n'.format(oyaml.dump(config_yaml)), 'green'))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.train.cuda_device
    cuda = False if config.train.nocuda else True

    resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on " + torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    now = datetime.datetime.now()
    date_str = now.strftime("%d-%m-%Y_%H-%M")

    t = []
    torch.manual_seed(329)

    bias = True if config.train.bias else False

    # build model
    model = SharpNet(ResBlock, [3, 4, 6, 3], [2, 2, 2, 2, 2],
                     use_normals=True if config.train.normals else False,
                     use_depth=True if config.train.depth else False,
                     use_boundary=True if config.train.boundary else False,
                     bias_decoder=bias)

    model_dict = model.state_dict()

    # Load pretrained weights

    resnet_path = 'models/resnet50-19c8e357.pth'

    if not os.path.exists(resnet_path):
        command = 'wget ' + resnet50_url + ' && mkdir models/ && mv resnet50-19c8e357.pth models/'
        os.system(command)

    resnet50_dict = torch.load(resnet_path)

    resnet_dict = {k.replace('.', '_img.', 1): v for k, v in resnet50_dict.items() if
                   k.replace('.', '_img.', 1) in model_dict}  # load weights up to pool

    print('Loading checkpoint from {}'.format(config.train.pretrained_model))
    if config.train.pretrained_model is not None:
        model_path = config.train.pretrained_model
        tmp_dict = torch.load(model_path)
        if config.train.depth:
            pretrained_dict = {k: v for k, v in tmp_dict.items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in tmp_dict.items() if
                               (k in model_dict and not k.startswith('depth_decoder'))}

    else:
        pretrained_dict = resnet_dict

    try:
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Successfully loaded pretrained ResNet weights')
    except:
        print('Could not load the pretrained model weights')
        sys.exit(0)

    # model = nn.DataParallel(model)
    model.to(device)
    model.zero_grad()
    model.train()

    freeze_decoders = config.train.decoder_freeze.split(',')
    freeze_model_decoders(model, freeze_decoders)

    if config.train.dataset != 'NYU':
        sharpnet_loss = SharpNetLoss(lamb=0.5, mu=1.0,
                                     use_depth=True if config.train.depth else False,
                                     use_boundary=True if config.train.boundary else False,
                                     use_normals=True if config.train.normals else False,
                                     use_geo_consensus=True if config.train.geo_consensus else False)
    else:
        sharpnet_loss = SharpNetLoss(lamb=0.5, mu=1.0,
                                     use_depth=True if config.train.depth else False,
                                     use_boundary=False,
                                     use_normals=False,
                                     use_geo_consensus=True if config.train.geo_consensus else False)

    if config.train.optimizer == 'SGD':
        optimizer = SGD(params=get_params(model),
                        lr=float(config.train.learning_rate),
                        weight_decay=float(config.train.decay),
                        momentum=0.9)
    elif config.train.optimizer == 'Adam':
        optimizer = Adam(params=get_params(model),
                         lr=float(config.train.learning_rate),
                         weight_decay=float(config.train.decay))
    else:
        print('Could not configure the optimizer, please select --optimizer Adam or SGD')
        sys.exit(0)

    # TensorBoard Logger
    train_loss_meter = MovingAverageValueMeter(20)
    val_loss_meter = MovingAverageValueMeter(3)
    depth_loss_meter = MovingAverageValueMeter(3) if config.train.depth else None
    normals_loss_meter = MovingAverageValueMeter(3) if config.train.normals and config.train.dataset != 'NYU' else None
    grad_loss_meter = MovingAverageValueMeter(3) if config.train.depth else None
    boundary_loss_meter = MovingAverageValueMeter(3) if config.train.boundary and config.train.dataset != 'NYU' else None
    consensus_loss_meter = MovingAverageValueMeter(3) if config.train.geo_consensus else None

    exp_name = config.train.experiment_name if config.train.experiment_name is not None else ''
    print('Experiment Name: {}'.format(exp_name))

    log_dir = os.path.join('logs', 'Joint', str(exp_name) + '_' + date_str)
    cp_dir = os.path.join('checkpoints', 'Joint', str(exp_name) + '_' + date_str)
    print('Checkpoint Directory: {}'.format(cp_dir))

    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val'))

    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(os.path.join(log_dir, 'train'))
        os.makedirs(os.path.join(log_dir, 'val'))

    train_dataloader, val_dataloader = get_trainval_splits(config)  # SHREK: Added Modification to pass in config.
                                                                # Either pass in path to config file, or real yaml and pass in the dict of config file.
                                                                # Config file need only contain the paths to datasets train and val.
                                                                # For val, we'd like to pass real images dataset.

    for epoch in range(config.train.max_epoch):
        if config.train.optimizer == 'SGD':
            adjust_learning_rate(float(config.train.learning_rate), config.train.lr_mode, float(config.train.gradient_step), config.train.max_epoch,
                                 optimizer, epoch)

        train_epoch(train_dataloader, val_dataloader, model, sharpnet_loss, optimizer, config.train.start_epoch + epoch,
                    train_writer, val_writer,
                    train_loss_meter, val_loss_meter,
                    depth_loss_meter, grad_loss_meter,
                    normals_loss_meter,
                    date_str=date_str, model_save_path=cp_dir,
                    config=config, boundary_loss_meter=boundary_loss_meter, consensus_loss_meter=consensus_loss_meter)

        # Save a model
        if epoch % 2 == 0 and epoch > int(0.9 * config.train.max_epoch):
            torch.save(
                model.state_dict(),
                os.path.join(cp_dir, 'checkpoint_{}_final.pth'.format(config.train.start_epoch + epoch)),
            )
        elif epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cp_dir, 'checkpoint_{}_final.pth'.format(config.train.start_epoch + epoch)),
            )
    torch.save(
        model.state_dict(),
        os.path.join(cp_dir, 'checkpoint_{}_final.pth'.format(config.train.start_epoch + config.train.max_epoch)),
    )

    return None


if __name__ == "__main__":
    main()
