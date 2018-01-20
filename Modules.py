# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:26:26 2017

@author: kalifou
"""

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

global CUDA
CUDA = True

def bi_normal(x1, x2, mu1, mu2, s1, s2, rho,M):

    pre_x1 = x1.contiguous().view(-1, 1).expand(x1.size(0), M)
    pre_x2 = x2.contiguous().view(-1, 1).expand(x1.size(0), M)
    if CUDA:
        pre_x2 = pre_x2.cuda()
        pre_x1 = pre_x1.cuda()
    x1 = Variable(pre_x1)
    x2 = Variable(pre_x2)
    #print type(x1),type(mu1)
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    z = torch.div(norm1, s1).pow(2) + torch.div(norm2, s2).pow(2) - 2 * torch.div(torch.mul(rho, torch.mul(norm1, norm2)), torch.mul(s1,s2))
    coef = torch.exp(-z/(2*(1-rho.pow(2))))
    #print(coef)
    denom = 2 * F.math.pi * s1 * s2 * torch.sqrt(1-rho.pow(2))
    #print(denom)
    result = torch.div(coef, denom)
    return result


def loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1, x2, pen, s, M, batch_size, N_max):
    
    indices = []
    for i in range(len(s)): indices += [a+250*i for a in range(s[i])]
    #indices = torch.LongTensor(indices)
    ls = bi_normal(x1[indices,], x2[indices,], z_mu1[indices,], z_mu2[indices,], z_sigma1[indices,], z_sigma2[indices,], z_corr[indices,],M)
    ls = torch.mul(ls, z_pi[indices,])
    ls = torch.sum(ls, 1, keepdim=True) # TODO change to Ns not Nmax
    ls = - torch.log(ls + 1e-6)
    #print "max :",ls.max()
    #print "dim ls :",ls.size(),x1.size()[0]
    ls = torch.sum(ls) / x1.size()[0]  
    lp = torch.log(z_pen_logits+ 1e-6)
    if CUDA:
        pen = pen.cuda()
    lp = torch.mul(lp.data, pen)
    lp = - torch.sum(lp, 1, keepdim=True)
    lp = torch.sum(lp) / x1.size()[0]
    
    
    return ls+lp

def Batch_Loss_LS_LP(X,Y_,M,N_max):
    
    l1,l2,l2 = Y_.size()
    Ls = 0.
    Lp = 0.
    for i in range(l1):
        for j in range(l2):
            Y = Y_[i,j,:]
            Pi_k, mu_x, mu_y, sig_x, sig_y, rho_xy = np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M)
            
            for m in range(M):
                Pi_k[m], mu_x[m], mu_y[m], sig_x[m], sig_y[m], rho_xy[m] =  Y[6*m:6*(m+1)].data.numpy()
                sig_x[m], sig_y[m], rho_xy[m] = np.exp(sig_x[m]), np.exp(sig_y[m]), np.tanh(rho_xy[m])
                 
            norm_Pi = np.exp(Pi_k).sum()
            Pi_k = np.exp(Pi_k)/norm_Pi
                
            q1,q2,q3 = Y[-3:]
            q1 = np.exp(q1)/np.exp(Y[-3:]).sum()
            q2 = np.exp(q2)/np.exp(Y[-3:]).sum()
            q3 = np.exp(q3)/np.exp(Y[-3:]).sum()    
            
            Sum_proba_multivariate = 0.
            Sum_proba_multivariate = np.sum( [ Pi_k[mt] * proba_x_y(X[i,j,0][0].data.numpy()[0],X[i,j,1][0].data.numpy()[0], mu_x[mt],mu_y[mt],sig_x[mt],sig_y[mt],rho_xy[mt]) for mt in range(M)])
            Ls += np.log(Sum_proba_multivariate)
            Lp += X[i,j,2].data.numpy()[0]*np.log(q1) \
                  +X[i,j,3].data.numpy()[0]*np.log(q2) \
                  +X[i,j,4].data.numpy()[0]*np.log(q3) 
            
    Ls = -Ls/N_max
    Lp = -Lp/N_max
    
    return Ls, Lp
    
def f_x_y(mu_x, mu_y, sig_x, sig_y, rho_xy):
    """returns (dX,dY)"""
    mean = [mu_x,mu_y]
    cov = [[sig_x*rho_xy, 0.], [0.,sig_y*rho_xy]]
    return np.random.multivariate_normal(mean,cov)

def proba_x_y(delta_x, delta_y, mu_x, mu_y, sig_x, sig_y, rho_xy):
    denom = 1./ (2 * F.math.pi* sig_x * sig_y * F.math.sqrt((1 - rho_xy**2)))
    coef = 1./ (2*(1-rho_xy**2))
    t_x = ((delta_x-mu_x)**2)/sig_x*sig_x
    t_y = ((delta_y-mu_y)**2)/sig_y*sig_y
    t_x_y = 2 * rho_xy * (delta_x - mu_x) * (delta_y-mu_y) / (sig_x*sig_y)
    e = F.math.exp(- coef *(t_x + t_y + t_x_y)  )  
    return denom * e       

class Encoder(nn.Module):
    
    def __init__(self, strokeSize, batchSize,  Nhe, Nhd, Nz):
        super(Encoder, self).__init__()
        self.Nz = Nz
        self.Nhe = Nhe
        self.Nhd = Nhd
        self.cell = nn.LSTM(strokeSize, Nhe//2, 1, bidirectional=True, batch_first=True)
        self.mu = nn.Linear(Nhe, Nz)
        self.sigma = nn.Linear(Nhe, Nz)
        self.h0 = nn.Linear(Nz, Nhd*2) # returns h0 and c0
        #print(Nh)

    def forward(self, x):
        _, (hn, cn) = self.cell(x)
        hn = Variable(torch.cat((hn.data[0],hn.data[1]),1))
        sigma = self.sigma(hn)
        mu = self.mu(hn)
        s = sigma
        sigma = torch.exp( sigma * 0.5)
        pre = torch.randn(self.Nz)
        if CUDA:
            pre = pre.cuda()
        eps = Variable(pre) 
        z = mu + sigma * eps
        return F.tanh(self.h0(z)), z, mu, s
        
        
class Decoder(nn.Module):
    
    def __init__(self, strokeSize, batchSize,  Nhe, Nhd, Nz, Ny):
        super(Decoder, self).__init__()
        self.Nhe = Nhe
        self.Nhd = Nhd
        self.cell = nn.LSTM(strokeSize+Nz, Nhd, 1, batch_first=True)
        self.y = nn.Linear(Nhd, Ny)
    
    def forward(self, x, h0, c0):
        #print('sizes :',h0.size(),c0.size())
        #print('sizes :',h0.size(),c0.size())
        h0 = h0.view(1, h0.size()[0], h0.size()[1])
        c0 = c0.view(1, c0.size()[0], c0.size()[1])
        output, (hn, cn) = self.cell(x, (h0, c0))
        #print('sizes :',h0.size(),c0.size())
        #print('sizes :',h0.size(),c0.size())
        #print("avant reshape", output.size())
        #print(output.contiguous().view(-1, self.Nh).size())
        output = output.contiguous().view(-1, self.Nhd)
        #print("after", output.size())
        y = self.y(output)
        return y
        
class SketchRNN(nn.Module):
    
    def __init__(self, strokeSize, batchSize, Nhe, Nhd, Nz, Ny, max_seq_len):
        super(SketchRNN, self).__init__()
        self.batchSize = batchSize
        self.Nhe = Nhe
        self.Nhd = Nhd
        self.max_seq_len = max_seq_len
        self.Nz = Nz
        self.encoder = Encoder(strokeSize, batchSize, Nhe, Nhd, Nz)
        self.decoder = Decoder(strokeSize, batchSize, Nhe, Nhd, Nz, Ny)
        
    def forward(self, x):
        # we don't take the inisiale S0 in the encoder so we do the 1:
        h0, z, mu, sigma = self.encoder(x[:, 1:250+1, :])
        #here we take S0
        new_input = torch.cat((x[:, :250, :], z.view(self.batchSize, 1, self.Nz)\
            .expand(self.batchSize, self.max_seq_len, self.Nz)), 2)        
        y = self.decoder(new_input, h0[:,:self.Nhd].contiguous(), h0[:,self.Nhd:].contiguous())
        #y = self.decoder(new_input, h0[:,:self.Nhd], h0[:,self.Nhd:])
        return y, mu, sigma
    
#class VAE(nn.Module):
#    
#    def __init__(self,input_size,output_size=128):
#        nn.Module.__init__(self)
#        self.input_size = input_size
#        self.output_size = output_size
#        
#        # Check sizes
#        variance = 1e-2
#        self.mu = nn.Linear(input_size,output_size)
#        self.mu.weight.data.uniform_(-variance,variance)
#        self.mu.bias.data.uniform_(-variance,variance)
#        
#        self.sigma = nn.Linear(input_size,output_size)
#        self.sigma.weight.data.uniform_(-variance,variance)
#        self.sigma.bias.data.uniform_(-variance,variance)
#        
#        self.h0 = nn.Linear(output_size,output_size)
#        self.h0 .weight.data.uniform_(-variance,variance)
#        self.h0 .bias.data.uniform_(-variance,variance)
#        
#    def forward(self, H):
#        sigma = torch.exp(self.sigma(H) / 2. )
#        mu = self.mu(H)
#        uniform_sampling = autograd.Variable(torch.FloatTensor(torch.randn(sigma.size())))        
#        z = sigma.mul(uniform_sampling)  + mu        
#        return F.tanh( self.h0(z)), z, mu, sigma
#      
#
#
#class Seq_2_Seq_VAE(nn.Module):
#    def __init__(self, obs_size, N_h, N_z, Decoder_size, Y_size, batch_size, max_seq_len):
#        nn.Module.__init__(self)
#        self.encoder = torch.nn.LSTM(obs_size , N_h/2, 1, batch_first=True, bidirectional=True)
#        self.vae = VAE(N_h,N_z)
#        self.decoder = torch.nn.LSTM(Decoder_size, N_h , 1, batch_first=True)
#        self.extra_layer = torch.nn.Linear(N_h,Y_size)
#        self.N_h = N_h
#        self.batch_size, self.max_seq_len, self.N_z = batch_size, max_seq_len, N_z
#    
#    def forward(self, X):
#        _,(h0,c0) = self.encoder(X) 
#        h1 = torch.cat((h0.data[0],h0.data[1]),1) # check that we concatened in the good sens
#        v_z, z, mu, sigma = self.vae(Variable(h1)) 
#        z = z.view(-1,self.batch_size * self.N_z).repeat(6,self.max_seq_len+1,1).data
#        X_decoder = torch.cat((X,z),2)
#        Y_pre, (h2,c2) = self.decoder(X_decoder) 
#        Y_pre = Y_pre.contiguous().view(-1, self.N_h)
#        Y_final = self.extra_layer(Y_pre)
#        
#        print "\nsize X ", X.size(),"\nsize H ", h0.size(),"\nSize Z ", v_z.size()
#        print "size X_Decoder", X_decoder.size(),"\nsize Y", Y_final.size(),"\n\n"
#        print "Y type :",type(Y_final.data.numpy()),Y_final.data.numpy().shape
#        
#        return mu, sigma, X_decoder,Y_final
