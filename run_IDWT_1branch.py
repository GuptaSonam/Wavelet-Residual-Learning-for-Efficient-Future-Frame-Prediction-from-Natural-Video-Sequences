__author__ = 'yunbo'

import os
import shutil
import argparse
import numpy as np
import torch
from core.data_provider import datasets_factory
from core.models.model_factory_IDWT_1branch import Model
from core.utils import preprocess
import core.trainer_IDWT_1branch as trainer
import torch.nn as nn
import pywt
#from pytorch_wavelets import DWTForward, DWTInverse

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cpu')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='/media/vplab/Sonam_HDD1/CVPR_Workshop_2020/FrequencyBasedModel/predrnn-pytorch-master/data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='/media/vplab/Sonam_HDD1/CVPR_Workshop_2020/FrequencyBasedModel/predrnn-pytorch-master/data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--sliding_window', type=int, default=3)
parser.add_argument('--wavelet_level', type=int, default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--logfile', type=str, default='')

args = parser.parse_args()
print(args)

def schedule_sampling(eta, itr):
    img_width = int(args.img_width/2)
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      img_width // args.patch_size,
                      img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_width // args.patch_size,
                    img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((img_width // args.patch_size,
                      img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            img_width // args.patch_size,
                            img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def train_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=True)

    eta = args.sampling_start_value

    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()                        #(8,20,64,64,1)
        ims_dwt= []
        #xfm = DWTForward(J=1, wave='bior1.3')
        # multilevel Haar wavelet transform
        for t in range(args.total_length):
            #print('t: ',t)
            inp = ims[:,t,:,:,:]
            #print('inp', inp.shape)
            inp = np.transpose(inp, (0,3,1,2))
            #print('lf', lf.shape)
            #X = torch.from_numpy(inp)
            #X = X.permute(0,3,1,2)
            #Yl, Yh = xfm(X)
            #Y = torch.cat((torch.squeeze(Yl), Yh), 1) 
            coeffs2 = pywt.dwt2(inp, 'haar')
            LL, (LH, HL, HH) = coeffs2
            #print('LL', LL.shape)
            inp_dwt = np.concatenate((LL, LH, HL, HH), 1)
            #print('inp_dwt', inp_dwt.shape)
            ims_dwt.append(inp_dwt)
        #ims_dwt = ims_dwt.detach().cpu().numpy()    
        ims_dwt = np.array(ims_dwt)
        #print('ims_dwt_before', ims_dwt.shape)
        ims_dwt = np.transpose(ims_dwt, (1,0,3,4,2))              #(8,3,128,128,7)
        #print('ims_dwt_before', ims_dwt.shape)  
        #print('ims_lf', ims_lf.shape)
        #print('ims_lf_process', ims_lf.shape)
        ims_dwt = preprocess.reshape_patch(ims_dwt, args.patch_size)        #(8,20,16,16,16)

        eta, real_input_flag = schedule_sampling(eta, itr)
        #print(ims_dwt.shape)
        trainer.train(model, ims_dwt, real_input_flag, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            trainer.test(model, test_input_handle,  args, itr)

        train_input_handle.next()


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')

'''
if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)
'''
#gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
#args.n_gpu = len(gpu_list)
print('Initializing models')

model = Model(args)
#model = nn.DataParallel(model, device_ids=[2, 3])

if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)

