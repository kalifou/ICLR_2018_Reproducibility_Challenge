# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:26:00 2018

@author: kalifou
"""
import numpy as np
import utils
import torch 
from torch.autograd import Variable
from Modules import VAE, Batch_Loss_LS_LP, proba_x_y, Seq_2_Seq_VAE
import torch.optim as optim
from matplotlib import pyplot as plt

filename = "sketch-rnn-datasets/aaron_sheep/aaron_sheep.npz"
load_data = np.load(filename)
train_set = load_data['train'][:100]
valid_set = load_data['valid'][:100]
test_set = load_data['test'][:100]

nb_steps = 10
feature_len=5
batch_size = 6
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

valid_set = utils.DataLoader(
      valid_set,
      batch_size,
      max_seq_length=max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
valid_set.normalize(normalizing_scale_factor)

test_set = utils.DataLoader(
      test_set,
      batch_size,
      max_seq_length=max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
test_set.normalize(normalizing_scale_factor)


M = 20
obs_size=5
Y_size = 6*M+3
N_h = 100
N_z = 128
X_decoder_size = N_z * batch_size + obs_size

s2s_vae = Seq_2_Seq_VAE(obs_size, N_h, N_z, X_decoder_size, Y_size, batch_size, max_seq_len)

w_lk = 1. # ?
L_kl =0.0
Ls = 0.
Lp = 0.
N_max = max_seq_len
lr=1e-2
optim = optim.Adam(s2s_vae.parameters(),lr) #to fill with missing optim params
L__=[]
for it in range(nb_steps):    
    _, x, s = train_set.random_batch()
    X = torch.from_numpy(x).type(torch.FloatTensor)
    mu, sigma, X_decoder,Y_final = s2s_vae(Variable(X))
    L_kl = -0.5 * torch.sum(1 + sigma- mu.pow(2) - sigma.exp())
    print it,"Loss KL : ",L_kl.data[0],'\n\n'
    
    l1,l2,l2 = Y_final.size()
    Ls,Lp = Batch_Loss_LS_LP(X_decoder,Y_final,M,N_max)
    print "Ls_Loss",Ls
    print "Lp_Loss",Lp
    Loss = (Ls+Lp) + w_lk * L_kl # Loss = lr + wlk * L_lk
    L__.append(L_kl.data[0])
    L_kl.backward()
    optim.step()

fig = plt.figure(figsize=(7,6))    
fig.suptitle("Loss KL",fontsize=15)
plt.plot(range(len(L__)),L__, 'b-')
plt.xlabel("Batches", fontsize=12)
plt.show()