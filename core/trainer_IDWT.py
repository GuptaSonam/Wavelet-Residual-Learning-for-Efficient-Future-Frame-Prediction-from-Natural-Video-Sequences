from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import datetime
import cv2
import numpy as np
from skimage.measure import compare_ssim
from core.utils import preprocess
import pywt
from PIL import Image


def batch_psnr(gen_frames, gt_frames):
  """Computes PSNR for a batch of data."""
  
  if gen_frames.ndim == 3:
    axis = (1, 2)
  elif gen_frames.ndim == 4:
    axis = (1, 2, 3)
  x = np.int32(gen_frames)
  y = np.int32(gt_frames)
  num_pixels = float(np.size(gen_frames[0]))
  mse = np.sum((x - y)**2, axis=axis, dtype=np.float32) / num_pixels
  if mse.all()==0:
    print('mse', mse)
  psnr = 20 * np.log10(255) - 10 * np.log10(mse)
  return np.mean(psnr)

def train(model, ims, ims_lf, ims_hf, real_input_flag, configs, itr):
    cost = model.train(ims, ims_lf, ims_hf, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        ims_lf = np.flip(ims_lf, axis=1).copy()
        ims_hf = np.flip(ims_hf, axis=1).copy()
        cost += model.train(ims_rev, ims_lf, ims_hf, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
         print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
         print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    fname = "logs" + "/" + configs.logfile + ".txt"
    fh = open(fname, "a")
    fh.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' test...' + str(itr) + '\n')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        mode = pywt.Modes.sp1DWT = 1 
        w = pywt.Wavelet('haar')
        ims_lf,ims_hf=[], []
        for t in range(configs.total_length - configs.sliding_window + 1):
            input = test_ims[:, t:t+configs.sliding_window,:,:,:]
            (lf, hf) = pywt.dwt(input, 'haar', axis=1)
            ims_lf.append(np.squeeze(lf))
            ims_hf.append(np.squeeze(hf))
        ims_lf = np.array(ims_lf)
        ims_hf = np.array(ims_hf)
        ims_lf_original = np.transpose(ims_lf, (1,0,3,4,2))              #(8,7,64,64,2)
        ims_hf_original = np.transpose(ims_hf, (1,0,3,4,2))              #(8,7,64,64,2)    
        
        #print('ims_lf_orig', ims_lf_original.shape)
        ims_lf_original_shape = ims_lf_original.shape
        ims_lf = preprocess.reshape_patch(ims_lf_original, configs.patch_size)
        #print(ims_lf.shape )
        ims_hf = preprocess.reshape_patch(ims_hf_original, configs.patch_size)   #(8,7,16,16,32)
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        
        img_gen_lf, img_gen_hf = model.test(test_dat, ims_lf, ims_hf, real_input_flag)         #()
        #print('img_gen_lf', img_gen_lf.shape)  
        img_gen_lf = preprocess.reshape_patch_back(img_gen_lf, configs.patch_size)            #(8, 16, 128, 128, 2)
        #print('img_gen_lf_reshape', img_gen_lf.shape)
        img_gen_hf = preprocess.reshape_patch_back(img_gen_hf, configs.patch_size)
        img_gen_lf = img_gen_lf.transpose(0, 1, 4, 2, 3)
        #print('img_gen_transpose', img_gen_lf.shape)
        img_gen_hf = img_gen_hf.transpose(0, 1, 4, 2, 3)
        #print(img_gen_lf.shape)   
        steps = img_gen_lf.shape[1]
        img_gen = []
        # using IDWT to reconstruct the images
        for t in range(steps):
            a = pywt.idwt(img_gen_lf[:,t], img_gen_hf[:,t], 'haar', axis=1)
            img_gen.append( a[:, configs.sliding_window-1]) 
        img_gen = np.expand_dims(np.array(img_gen), axis=4)
        img_gen = img_gen.transpose(1,0,2,3,4) 
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score
            psnr[i] += batch_psnr(real_frm, pred_frm)

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length - configs.sliding_window + 1):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
                '''
                # input DWT images
                name = 'dwt_lf1_gt' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                #print('ims_lf_orig', ims_lf_original[0,i,:,:,0].shape)
                img_dwt_gt_lf = Image.fromarray(ims_lf_original[0,i,:,:,0])
                img_dwt_gt_lf.save(file_name)
                name = 'dwt_lf2_gt' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                img_dwt_gt_lf = Image.fromarray(ims_lf_original[0,i,:,:,1])
                img_dwt_gt_lf.save(file_name)
               
                name = 'dwt_hf1_gt' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                img_dwt_gt_hf = Image.fromarray(ims_hf_original[0,i,:,:,0])
                img_dwt_gt_hf.save(file_name)
                name = 'dwt_hf2_gt' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                img_dwt_gt_hf = Image.fromarray(ims_hf_original[0,i,:,:,1])
                img_dwt_gt_hf.save(file_name)
                '''
                
            for i in range(img_gen_length):
                '''
                # predicted DWTs 
                name = 'dwt_lf1_pd' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                #print('ims_lf_orig', ims_lf_original[0,i,:,:,0].shape)
                img_dwt_gt_lf = Image.fromarray(img_gen_lf[0,i,0,:,:])
                img_dwt_gt_lf.save(file_name)
                name = 'dwt_lf2_pd' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                img_dwt_gt_lf = Image.fromarray(img_gen_lf[0,i,1,:,:])
                img_dwt_gt_lf.save(file_name)

                name = 'dwt_hf1_pd' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                img_dwt_gt_hf = Image.fromarray(img_gen_hf[0,i,0,:,:])
                img_dwt_gt_hf.save(file_name)
                name = 'dwt_hf2_pd' + str(i + 1) + '.tif'
                file_name = os.path.join(path, name)
                img_dwt_gt_hf = Image.fromarray(img_gen_lf[0,i,0,:,:])
                img_dwt_gt_hf.save(file_name)
                '''
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()
    
    

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    fh.write('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))
        
    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    fh.write('\n ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    fh.write('\npsnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])
    fh.close()   
