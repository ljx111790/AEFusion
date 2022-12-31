# -*- coding:utf-8 -*-

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import AEFusion_autoencoder
from args_fusion import args
import pytorch_msssim
from tensorboardX import SummaryWriter
from loss_network import LossNetwork
import torch.nn as nn

device = 'cuda'


def main():
    original_imgs_path = utils.list_images(args.dataset)
    train_num = 80000
    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)
    for i in range(2, 3):
        # i = 2
        train(i, original_imgs_path)


def train(i, original_imgs_path):
    batch_size = args.batch_size

    input_nc = 1
    output_nc = 1

    deepsupervision = False
    nb_filter = [64, 112, 160, 208, 256]

    nest_model = AEFusion_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)

    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        nest_model.load_state_dict(torch.load(args.resume))
    # print(nest_model)
    optimizer = Adam(nest_model.parameters(), args.lr)
    # L1_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    ssim_loss = pytorch_msssim.msssim

    nest_model.cuda()

    tbar = trange(args.epochs)
    # tbar = tqdm(range(args.epochs))

    print('Start training.....')

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_pixel_loss = 0.

    MSE_fun = nn.MSELoss()
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()

    # plotfeater=True
    # writer = SummaryWriter(log_dir='logs', flush_secs=160)


    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        nest_model.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
            count += 1
            optimizer.zero_grad()
            img = Variable(img, requires_grad=False)
            if args.cuda:
                img = img.cuda()
            # get fusion image
            # encoder
            en = nest_model.encoder(img)
            # decoder
            outputs = nest_model.decoder_train(en)
            # resolution loss: between fusion image and visible image
            x = Variable(img.data.clone(), requires_grad=False)

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output in outputs:
                pixel_loss_temp = L1_loss(output, x)
                ssim_loss_temp = ssim_loss(output, x, normalize=True)
                ssim_loss_value += (1 - ssim_loss_temp)
                pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            with torch.no_grad():
                x = img.detach()
            features = loss_network(x)

            features_re = loss_network(outputs[0])

            with torch.no_grad():
                f_x_vi1 = features[1].detach()
                f_x_vi2 = features[2].detach()
                f_x_ir3 = features[3].detach()
                f_x_ir4 = features[4].detach()

            perceptual_loss = MSE_fun(features_re[1], f_x_vi1) + MSE_fun(features_re[2], f_x_vi2) + \
                              MSE_fun(features_re[3], f_x_ir3) + MSE_fun(features_re[4], f_x_ir4)

            perceptual_loss = perceptual_loss * 1000

            # total loss
            total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value + perceptual_loss

            total_loss.backward()
            optimizer.step()

            # writer.add_scalar('total loss', total_loss, (batch))

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t   perceptual_loss: {:.6f}\t  total: {:.6f}".format(
                    time.ctime(), i, e + 1, count, batches, all_pixel_loss / args.log_interval,
                                     (args.ssim_weight[i] * all_ssim_loss) / args.log_interval, perceptual_loss.item(),
                                     (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
                )
                tbar.set_description(mesg)
                # print(mesg)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
                count_loss = count_loss + 1
                all_ssim_loss = 0.
                all_pixel_loss = 0.

            if (batch + 1) % (100 * args.log_interval) == 0:
                # save model
                print("save")
                nest_model.eval()
                nest_model.cpu()
                save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
                                          i] + ".model"
                save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
                print(save_model_path)
                torch.save(nest_model.state_dict(), save_model_path)
                # save loss data
                if args.save_train_loss_or_not:
                    # pixel loss
                    loss_data_pixel = Loss_pixel
                    loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
                        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                                  '_') + "_" + \
                                         args.ssim_path[i] + ".mat"
                    scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
                    # SSIM loss
                    loss_data_ssim = Loss_ssim
                    loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
                        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                                  '_') + "_" + \
                                         args.ssim_path[i] + ".mat"
                    scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
                    # all loss
                    loss_data = Loss_all
                    loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(
                        e) + "_iters_" + \
                                         str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                        '_') + "_" + \
                                         args.ssim_path[i] + ".mat"
                    scio.savemat(loss_filename_path, {'loss_all': loss_data})

                nest_model.train()
                nest_model.cuda()

                print("\nCheckpoint, trained model saved at", save_model_path)
    # pixel loss
    if args.save_Final_loss_or_not:
        loss_data_pixel = Loss_pixel
        loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
            args.epochs) + "_" + str(
            time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
        scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
        loss_data_ssim = Loss_ssim
        loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
            args.epochs) + "_" + str(
            time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
        scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
        # SSIM loss
        loss_data = Loss_all
        loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
            args.epochs) + "_" + str(
            time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
        scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
    # save model
    nest_model.eval()
    nest_model.cpu()
    save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
    save_model_path = os.path.join(args.Final_model_dir, save_model_filename)
    torch.save(nest_model.state_dict(), save_model_path)
    print('\ndone,trined checkpoint model save at', save_model_path)
    print("\nDone, trained model saved at", args.Final_model_dir)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
