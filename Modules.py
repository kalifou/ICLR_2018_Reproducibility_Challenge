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
    
class VAE(nn.Module):
    
    def __init__(self,input_size,output_size=128):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        
        # Check sizes
        variance = 1e-2
        self.mu = nn.Linear(input_size,output_size)
        self.mu.weight.data.uniform_(-variance,variance)
        self.mu.bias.data.uniform_(-variance,variance)
        
        self.sigma = nn.Linear(input_size,output_size)
        self.sigma.weight.data.uniform_(-variance,variance)
        self.sigma.bias.data.uniform_(-variance,variance)
        
        self.h0 = nn.Linear(output_size,output_size)
        self.h0 .weight.data.uniform_(-variance,variance)
        self.h0 .bias.data.uniform_(-variance,variance)
        
    def forward(self, H):
        sigma = torch.exp(self.sigma(H) / 2. )
        mu = self.mu(H)
        uniform_sampling = autograd.Variable(torch.FloatTensor(torch.randn(sigma.size())))        
        z = sigma.mul(uniform_sampling)  + mu        
        return F.tanh( self.h0(z)), z, mu, sigma
      


class Seq_2_Seq_VAE(nn.Module):
    def __init__(self, obs_size, N_h, N_z, Decoder_size, Y_size, batch_size, max_seq_len):
        nn.Module.__init__(self)
        self.encoder = torch.nn.LSTM(obs_size , N_h/2, 1, batch_first=True, bidirectional=True)
        self.vae = VAE(N_h,N_z)
        self.decoder = torch.nn.LSTM(Decoder_size, N_h , 1, batch_first=True)
        self.extra_layer = torch.nn.Linear(N_h,Y_size)
        
        self.batch_size, self.max_seq_len, self.N_z = batch_size, max_seq_len, N_z
    
    def forward(self, X):
        _,(h0,c0) = self.encoder(X) 
        h1 = torch.cat((h0.data[0],h0.data[1]),1) # check that we concatened in the good sens
        v_z, z, mu, sigma = self.vae(Variable(h1)) 
        z = z.view(-1,self.batch_size * self.N_z).repeat(6,self.max_seq_len+1,1).data
        X_decoder = torch.cat((X,z),2)
        Y_pre, (h2,c2) = self.decoder(X_decoder) 
        Y_final = self.extra_layer(Y_pre)
        
        print "\nsize X ", X.size(),"\nsize H ", h0.size(),"\nSize Z ", v_z.size()
        print "size X_Decoder", X_decoder.size(),"\nsize Y", Y_final.size(),"\n\n"
        print "Y type :",type(Y_final.data.numpy()),Y_final.data.numpy().shape
        
        return mu, sigma, X_decoder,Y_final