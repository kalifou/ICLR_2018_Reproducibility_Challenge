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

early_stopping = False
reload_ = False
cuda = True
M = 20
obs_size=5
Y_size = 6*M+3
N_he = 512 #4
N_hd = 512
N_z = 128
w_lk = 0.5
lr=1e-4
N_max = max_seq_len
X_decoder_size = N_z * batch_size + obs_size

if reload_:
    s2s_vae = pickle.load(open('sketch_rnn_save.p', 'rb'))
    print("Reload")
else :
                        #strokeSize, batchSize, Nhe, Nhd, Nz, Ny, max_seq_len
    s2s_vae = SketchRNN(obs_size, batch_size, N_he, N_hd, N_z, 6*M+3, max_seq_len)
if cuda:
    s2s_vae.encoder.cuda()
    s2s_vae.decoder.cuda()
    s2s_vae.cuda()
    cudnn.benchmark = True
    print("Using cuda")

if early_stopping:
    print("Using early stopping")
    
optim = optim.Adam(s2s_vae.parameters(),lr) #to fill with missing optim params

Lr_train=[]
Lk_train =[]
L_train_early_stopping =[]
L_train=[]

Lr_test=[]
Lk_test =[]
L_test=[]
L_test_early_stopping=[]


for it in range(nb_steps):
    optim.zero_grad()  

    ################################
    # On Train set
    ################################
    _, x_train, s = train_set.random_batch()    
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    input = x_train
    if cuda:
        input = input.cuda()
    input = Variable(input)
    y, mu, sigma = s2s_vae(input)
    z_pen_logits = y[:, -3:]
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y[:, :-3], 6, dim=1)
    z_pi = F.softmax(z_pi)
    z_pen_logits = F.softmax(z_pen_logits)
    z_sigma1 = torch.exp(z_sigma1)
    z_sigma2 = torch.exp(z_sigma2)
    z_corr = F.tanh(z_corr)
    targets = x_train[:, 1:250+1, :].contiguous().view(-1, 5)
    x1 = targets[:,0]
    x2 = targets[:,1]
    pen = targets[:,2:]
   
    lr =Lr(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1, x2, pen, s, M, batch_size, max_seq_len)
    #print("lr" ,lr.data[0])
    L_kl = Lkl(mu, sigma)
    L_kl=  (L_kl + 1e-6) # / float(N_z) #*batch_size)
    
    #print('losses :',lr.data[0]," , ",L_kl.data[0])
    Lr_train.append(lr.data[0])
    Lk_train.append(L_kl.data[0])
    #L_train_early_stopping.append(early_stopping_Loss(lr,w_lk,L_kl, it).data[0])

    if early_stopping :
        L_train.append(early_stopping_Loss(lr,w_lk,L_kl, it).data[0])
        early_stopping_Loss(lr,w_lk,L_kl, it).backward()
    else:            
        L_train.append(lr.data[0]+w_lk * L_kl.data[0])
        (lr + w_lk * L_kl).backward()

    optim.step()
    
    ################################
    # On Test Set
    ################################
    _, x_test, s = test_set.random_batch()    
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    input = x_test
    if cuda:
        input = input.cuda()
    input = Variable(input)
    y, mu, sigma = s2s_vae(input)
    z_pen_logits = y[:, -3:]
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y[:, :-3], 6, dim=1)
    z_pi = F.softmax(z_pi)
    z_pen_logits = F.softmax(z_pen_logits)
    z_sigma1 = torch.exp(z_sigma1)
    z_sigma2 = torch.exp(z_sigma2)
    z_corr = F.tanh(z_corr)
    targets = x_test[:, 1:250+1, :].contiguous().view(-1, 5)
    x1 = targets[:,0]
    x2 = targets[:,1]
    pen = targets[:,2:]
    
    lr = Lr(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1, x2, pen, s, M, batch_size, max_seq_len)    
    L_kl = Lkl(mu, sigma)
    L_kl=  (L_kl + 1e-6) #/ float(N_z*batch_size)
    
    Lr_test.append(lr.data[0])
    Lk_test.append(L_kl.data[0])
    #L_test_early_stopping.append(early_stopping_Loss(lr,w_lk,L_kl, it).data[0])

    #if early_stopping :
    #    L_test.append(early_stopping_Loss(lr,w_lk,L_kl, it).data[0])
    #    early_stopping_Loss(lr,w_lk,L_kl, it).backward()
    #else:            
    #    L_test.append(lr.data[0]+w_lk * L_kl.data[0])
    #    (lr + w_lk * L_kl).backward()    

    if ((it+1)%10)==0:
        print("Iter :",it)
        print('losses :',lr.data[0]," , ",L_kl.data[0])
        Lk_train = np.nan_to_num(Lk_train)
        Lr_train = np.nan_to_num(Lr_train)
        L_train  = np.nan_to_num(L_train)
        
        Lk_test = np.nan_to_num(Lk_test)
        Lr_test = np.nan_to_num(Lr_test)
        L_test  = np.nan_to_num(L_test)
        #x_range = [ 1+i for i in range(len(Lk))]
        fig = plt.figure(figsize=(7,6))    

        fig.suptitle("Loss KL",fontsize=15)
        plt.plot(range(len(Lk_test)),Lk_test, 'b-',label="test")
        plt.plot(range(len(Lk_train)),Lk_train, 'r-',label="train")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.xlabel("Batches", fontsize=12)
        plt.savefig("tmp/L_lk.png")

        fig = plt.figure(figsize=(7,6))    
        fig.suptitle("Loss Lr",fontsize=15)
        plt.plot(range(len(Lr_test)),Lr_test, 'b-',label="test")
        plt.plot(range(len(Lr_train)),Lr_train, 'r-',label="train")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.xlabel("Batches", fontsize=12)
        plt.savefig("tmp/L_lr.png")


        fig = plt.figure(figsize=(7,6))    
        fig.suptitle("L_r + w_lk x L_kl",fontsize=15)
        plt.plot(range(len(L_test)),L_test, 'b-',label="test")
        plt.plot(range(len(L_train)),L_train, 'r-',label="train")
        #plt.plot(range(len(L_train_early_stopping)),L_train_early_stopping, 'g-',label="Early_train ")
        #plt.plot(range(len(L_test_early_stopping)),L_test_early_stopping, 'y-',label="Early_test ")
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.xlabel("Batches", fontsize=12)
        plt.savefig("tmp/L_total.png")
        Lk_test = Lk_test.tolist()
        Lr_test = Lr_test.tolist()
        L_test = L_test.tolist()
        
        Lk_train = Lk_train.tolist()
        Lr_train = Lr_train.tolist()
        L_train = L_train.tolist()
            
    if ((it+1) % 2000) == 0:
        print("Iter :",it," saving model")
        pickle.dump(s2s_vae, open('sketch_rnn_save.p', 'wb'))


t2 = time.time()-t1
print("Exec duration(s) : ",t2)
pickle.dump(s2s_vae, open('sketch_rnn_save.p', 'wb'))

print("Running & saving model duration(s) : ",time.time()-1)



