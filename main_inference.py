# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:26:00 2018

@author: kalifou
"""
import numpy as np
import utils
import torch 
from torch.autograd import Variable
from Modules import  SketchRNN, Lr, Lkl, early_stopping_Loss
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import _pickle as pickle

t1 = time.time()
filename = "sketch-rnn-datasets/aaron_sheep/aaron_sheep.npz"
load_data = np.load(filename, encoding = 'latin1')
train_set = load_data['train']
valid_set = load_data['valid']
test_set = load_data['test']

nb_steps = 10000
feature_len=5
batch_size = 1
max_seq_len=250

augment_stroke_prob=0.1
random_scale_factor=0.1

train_set = utils.DataLoader(
      train_set,
      batch_size,
      max_seq_length=max_seq_len,
      random_scale_factor=random_scale_factor,
      augment_stroke_prob=augment_stroke_prob)

normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
train_set.normalize(normalizing_scale_factor)


reload_ = True
cuda = True
M = 20
obs_size=5
Y_size = 6*M+3
N_he = 512 #4
N_hd = 512
N_z = 128
w_lk = 0.5
lr=1e-3
N_max = max_seq_len
X_decoder_size = N_z * batch_size + obs_size

if reload_:
    s2s_vae = pickle.load(open('sketch_rnn_save_80000.p', 'rb'))
    s2s_vae.decoder.Nz=N_z
    s2s_vae.decoder.batchSize = batch_size
    print("Model reloaded")

else :
                        #strokeSize, batchSize, Nhe, Nhd, Nz, Ny, max_seq_len
    s2s_vae = SketchRNN(obs_size, batch_size, N_he, N_hd, N_z, 6*M+3, max_seq_len)

if cuda:
    #s2s_vae.encoder.cuda()
    #s2s_vae.decoder.cuda()
    s2s_vae.cuda()
    cudnn.benchmark = True
    print("Using cuda")

for _ in range(10):
    _, x, s = train_set.random_batch()
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda())
    s2s_vae.predict(x, M)
