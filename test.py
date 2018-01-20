# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:26:00 2018

@author: kalifou
"""
import numpy as np
import utils
import torch 
from torch.autograd import Variable
from Modules import Batch_Loss_LS_LP, proba_x_y,bi_normal,loss, SketchRNN
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
batch_size = 100
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

reload_ = False
cuda = True
M = 20
obs_size=5
Y_size = 6*M+3
N_h = 512 #4
N_z = 128
w_lk = 0.5
lr=1e-3
N_max = max_seq_len
X_decoder_size = N_z * batch_size + obs_size

if reload_:
    s2s_vae = pickle.load(open('sketch_rnn_save.p', 'rb'))
    print("Reload")
else :
    s2s_vae = SketchRNN(obs_size, batch_size,N_h, N_z, 6*M+3, max_seq_len)
if cuda:
    s2s_vae.encoder.cuda()
    s2s_vae.decoder.cuda()
    s2s_vae.cuda()
    cudnn.benchmark = True
    print("Using cuda")

optim = optim.Adam(s2s_vae.parameters(),lr) #to fill with missing optim params
Lr=[]
Lk =[]
L_=[]

for it in range(nb_steps):
    optim.zero_grad()    
    _, x, s = train_set.random_batch()
    
    x = torch.from_numpy(x).type(torch.FloatTensor)
    input = x
    if cuda:
        input = input.cuda()
    input = Variable(input)
    y, y_bis, mu, sigma = s2s_vae(input)
    z_pen_logits = y[:, -3:]
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y[:, :-3], 6, dim=1)
    z_pi = F.softmax(z_pi)
    z_pen_logits = F.softmax(z_pen_logits)
    z_sigma1 = torch.exp(z_sigma1)
    z_sigma2 = torch.exp(z_sigma2)
    z_corr = F.tanh(z_corr)
    targets = x[:, 1:250+1, :].contiguous().view(-1, 5)
    x1 = targets[:,0]
    x2 = targets[:,1]
    pen = targets[:,2:]
    lr = loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1, x2, pen, s, M, batch_size, max_seq_len)
    L_kl = -0.5 * torch.sum(1 + sigma- mu.pow(2) - sigma.exp()+1e-6) 
    #print 'size LKL :',(1 + sigma- mu.pow(2) - sigma.exp()).size(),N_z
    L_kl=  (L_kl + 1e-6) / float(N_z*batch_size)
    #print('losses :',lr.data[0]," , ",L_kl.data[0])
    Lr.append(lr.data[0])
    Lk.append(L_kl.data[0])
    L_.append(lr.data[0]+w_lk * L_kl.data[0])
    (lr + w_lk * L_kl).backward()
    optim.step()

    if (it%200)==0:
        print('losses :',lr.data[0]," , ",L_kl.data[0])

    if ((it+1) % 2000) == 0:
        print("Iter :",it," saving model")
        pickle.dump(s2s_vae, open('sketch_rnn_save.p', 'wb'))

Lk = np.nan_to_num(Lk)
Lr = np.nan_to_num(Lr)
L_ = np.nan_to_num(L_)
#x_range = [ 1+i for i in range(len(Lk))]
fig = plt.figure(figsize=(7,6))    

fig.suptitle("Loss KL",fontsize=15)
plt.plot(range(len(Lk)),Lk, 'b-')
plt.xlabel("Batches", fontsize=12)
plt.savefig("plots/L_lk.png")

fig = plt.figure(figsize=(7,6))    
fig.suptitle("Loss Lr",fontsize=15)
plt.plot(range(len(Lr)),Lr, 'r-')
plt.xlabel("Batches", fontsize=12)
plt.savefig("plots/L_lr.png")


fig = plt.figure(figsize=(7,6))    
fig.suptitle("L_r + w_lk x L_kl",fontsize=15)
plt.plot(range(len(L_)),L_, 'g-')
plt.xlabel("Batches", fontsize=12)
plt.savefig("plots/L_total.png")

t2 = time.time()-t1
print("Exec duration(s) : ",t2)
pickle.dump(s2s_vae, open('sketch_rnn_save.p', 'wb'))

print("Running & saving model duration(s) : ",time.time()-1)



