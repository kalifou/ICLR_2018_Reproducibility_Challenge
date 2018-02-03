import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import os
import utils
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
import time
import _pickle as pickle
import PIL



def bi_normal(x1, x2, mu1, mu2, s1, s2, rho,M):

    x1 = x1.contiguous().view(-1, 1).expand(x1.size(0), M)
    x2 = x2.contiguous().view(-1, 1).expand(x1.size(0), M)
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    z = torch.div(norm1, s1).pow(2) + torch.div(norm2, s2).pow(2) - 2 * torch.div(torch.mul(rho, torch.mul(norm1, norm2)), torch.mul(s1,s2))
    coef = torch.exp(-z/(2*(1-rho.pow(2))))
    denom = 2 * F.math.pi * s1 * s2 * torch.sqrt(1-rho.pow(2))
    result = torch.div(coef, denom)
    return result

def Lr(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1, x2, pen, s, M, batch_size, N_max):
    
    indices = []
    for i in range(len(s)): indices += [a+200*i for a in range(s[i])]
    #indices = torch.LongTensor(indices)
    ls = bi_normal(x1[indices,], x2[indices,], z_mu1[indices,], z_mu2[indices,], z_sigma1[indices,], z_sigma2[indices,], z_corr[indices,],M)
    ls = torch.mul(ls, z_pi[indices,])
    ls = torch.sum(ls, 1, keepdim=True)
    ls = - torch.log(ls + 1e-6)
    ls = torch.sum(ls) / x1.size()[0]
                   
    lp = torch.log(z_pen_logits + 1e-6)

    lp = torch.mul(lp, pen)
    lp = - torch.sum(lp, 1, keepdim=True)
    lp = torch.sum(lp) / x1.size()[0]
    return ls+lp

def Lkl(mu, sigma,  R=0.99999, KL_min = 0.2):
    eta_step = 1 - (1- 0.01) * R
    Lkl = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())                 
    return 0.5 * eta_step * torch.max(Lkl, Variable(torch.FloatTensor([0.2])).cuda())
    

def make_image(sequence,epoch, name='_output_'):
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())
    name = str(epoch)+name+'.jpg'
    pil_image.save(name,"JPEG")
    plt.close("all")


class Encoder(nn.Module):
    
    def __init__(self, strokeSize, batchSize,  Nhe, Nhd, Nz,dropout=0.1):
        super(Encoder, self).__init__()
        self.Nz = Nz
        self.Nhe = Nhe
        self.Nhd = Nhd
        self.cell = nn.LSTM(strokeSize, Nhe, 1, dropout=dropout, bidirectional=True, batch_first=True)
        self.mu = nn.Linear(Nhe*2, Nz)
        self.sigma = nn.Linear(Nhe*2, Nz)
        self.h0 = nn.Linear(Nz, Nhd*2) # returns h0 and c0
        self.train()
        #print(Nh)

    def forward(self, x):
        _, (hn, cn) = self.cell(x)
        hn = Variable(torch.cat((hn.data[0],hn.data[1]),1)).cuda()
        sigma = self.sigma(hn)
        mu = self.mu(hn)
        s = sigma
        sigma = torch.exp( sigma * 0.5)
        N = Variable(torch.normal(torch.zeros(self.Nz),torch.ones(self.Nz))).cuda()
        z = mu + sigma * N
        return F.tanh(self.h0(z)), z, mu, s
        
        
class Decoder(nn.Module):
    
    def __init__(self, strokeSize, batchSize,  Nhe, Nhd, Nz, Ny,dropout=0.1):
        super(Decoder, self).__init__()
        self.Nhe = Nhe
        self.Nhd = Nhd
        self.Nz = Nz
        self.batchSize = batchSize
        self.cell = nn.LSTM(strokeSize+Nz, Nhd, 1, dropout=dropout, batch_first=True)
        self.y = nn.Linear(Nhd, Ny)
    
    def forward(self, x, h0, c0):
        h0 = h0.view(1, h0.size()[0], h0.size()[1])
        c0 = c0.view(1, c0.size()[0], c0.size()[1])
        output, (hn, cn) = self.cell(x, (h0, c0))
        output = output.contiguous().view(-1, self.Nhd)
        y = self.y(output)
        return y
                   
    def predict(self, x, h, c, z, M):
        out = []
        tmp = 0.2
        new_input = torch.cat((x[:, 0, :].contiguous().view(1, 1, 5), z.view(1, 1, self.Nz)), 2)

        h = h.view(1, h.size()[0], h.size()[1])
        c = c.view(1, c.size()[0], c.size()[1])
        
        for i in range(0,200):
            y, (h, c) = self.cell(new_input, (h, c))
            output = y.contiguous().view(-1, self.Nhd)
            y = self.y(output)
            z_pen_logits = y[:, -3:]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y[:, :-3], 6, dim=1)
            z_pi = F.softmax(z_pi / tmp)
            z_pen_logits = F.softmax(z_pen_logits / tmp)
            z_sigma1 = torch.exp(z_sigma1)
            z_sigma2 = torch.exp(z_sigma2)
            z_corr = F.tanh(z_corr)

            probs = z_pi.data[0].cpu().numpy()
            probs /= probs.sum()          
            i = np.random.choice(np.arange(0, M), p=probs)
            mean = [z_mu1.data[0][i], z_mu2.data[0][i]]
            cov = [[z_sigma1.data[0][i] * z_sigma1.data[0][i] * tmp, z_corr.data[0][i] * z_sigma1.data[0][i] * z_sigma2.data[0][i]], \
                   [z_corr.data[0][i] * z_sigma1.data[0][i] * z_sigma2.data[0][i], z_sigma2.data[0][i] * z_sigma2.data[0][i] * tmp]]

            x = np.random.multivariate_normal(mean, cov)
            probs = z_pen_logits.data[0].cpu().numpy()
            probs /= probs.sum()
            iPen = np.random.choice(np.arange(0, 3), p=probs)
            pen = [0, 0, 0]
            pen[iPen] = 1
        
            stroke = [x[0], x[1]] + pen
            print(stroke)
            
            if iPen == 2:
                break
            
            out.append(stroke)

            stroke = torch.FloatTensor(stroke)
            stroke = Variable(stroke).cuda()
            new_input = torch.cat((stroke.contiguous().view(1, 1, 5), z.view(1, 1, self.Nz).data), 2)            
            
        return out


def infere(encoder, decoder, load_data):
    
    inference_set = load_data['train']
    inference_set = utils.DataLoader(
          inference_set,
          1,
          max_seq_length=max_seq_len)
    normalizing_scale_factor = inference_set.calculate_normalizing_scale_factor()
    inference_set.normalize(normalizing_scale_factor)
    _, x, s = inference_set.random_batch() 
    
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).cuda()
    
    encoder.train(False)
    decoder.train(False)
    h0, z, mu, sigma = encoder(x[:, 1:200+1, :])
    y = decoder.predict(x[:, :200, :].contiguous(), h0[:,:N_hd].contiguous(), h0[:,N_hd:].contiguous(), z, M)
    seq_x = [a[0] for a in y]
    seq_y = [a[1] for a in y]
    seq_z = [a[2]==0 for a in y]
    x_sample = np.cumsum(seq_x, 0)
    y_sample = np.cumsum(seq_y, 0)
    z_sample = np.array(seq_z)
    sequence = np.stack([x_sample,y_sample,z_sample]).T
    #!!!!!NOTE THAT 1 IS THE NAME OF THE GENERATED IMAGE 
    make_image(sequence,1)

    
if __name__=="__main__":
    
    t1 = time.time()
    filename = "cat.npz"
    load_data = np.load(filename, encoding = 'latin1')
    train_set = load_data['train']
    #valid_set = load_data['valid']
    #test_set = load_data['test']
    
    nb_steps = 20000
    batch_size = 100
    max_seq_len=200
    M = 20
    strokeSize=5
    Y_size = 6*M+3
    N_he = 256
    N_hd = 512
    N_z = 128
    w_lk = 0.5
    lr = 0.001
    temperature = 0.1
    lr_decay = 0.9999
    min_lr = 0.00001
    KL_min = 0.2
    grad_clipping = 1.0
    N_y = 6*M+3
    
    train_set = utils.DataLoader(
          train_set,
          batch_size,
          max_seq_length=max_seq_len)
    normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    train_set.normalize(normalizing_scale_factor)
    
    
    encoder = Encoder(strokeSize, batch_size, N_he, N_hd, N_z).cuda()
    decoder = Decoder(strokeSize, batch_size, N_he, N_hd, N_z, N_y).cuda()
    
    saved_encoder = torch.load('encoderOUR_epoch_9999.pth')
    saved_decoder = torch.load('decoderOUR_epoch_9999.pth')
    encoder.load_state_dict(saved_encoder)
    decoder.load_state_dict(saved_decoder)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr)
    
    cudnn.benchmark = True

    for it in range(10001, nb_steps):
        '''
        encoder_optimizer.zero_grad() 
        decoder_optimizer.zero_grad()
        
        _, x_train, s = train_set.random_batch()    
        x_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor)).cuda()
        
        h0, z, mu, sigma = encoder(x_train[:, 1:200+1, :])
        
        new_input = torch.cat((x_train[:, :200, :], z.view(batch_size, 1, N_z)\
                .expand(batch_size, max_seq_len, N_z)), 2)        
        y = decoder(new_input, h0[:,:N_hd].contiguous(), h0[:,N_hd:].contiguous())
    
    
        z_pen_logits = y[:, -3:]
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y[:, :-3], 6, dim=1)
        z_pi = F.softmax(z_pi)
        z_pen_logits = F.softmax(z_pen_logits)
        z_sigma1 = torch.exp(z_sigma1)
        z_sigma2 = torch.exp(z_sigma2)
        z_corr = F.tanh(z_corr)
        targets = x_train[:, 1:200+1, :].contiguous().view(-1, 5)
        x1 = targets[:,0]
        x2 = targets[:,1]
        pen = targets[:,2:]
       
        lr = Lr(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1, x2, pen, s, M, batch_size, max_seq_len)
        L_kl = Lkl(mu, sigma)
        L_kl=  (L_kl + 1e-6)
               
        loss = lr + L_kl
        loss.backward()
    
        # gradient cliping
        nn.utils.clip_grad_norm(encoder.parameters(), grad_clipping)
        nn.utils.clip_grad_norm(decoder.parameters(), grad_clipping)
        
        decoder_optimizer.step()
        encoder_optimizer.step()
        
        print('epoch',it,'loss',loss.data[0],'LR',lr.data[0],'LKL',L_kl.data[0])
        
        '''
        
        encoder = Encoder(strokeSize, batch_size, N_he, N_hd, N_z).cuda()
        decoder = Decoder(strokeSize, batch_size, N_he, N_hd, N_z, N_y).cuda()
        
        saved_encoder = torch.load('encoderOUR_epoch_19999.pth')
        saved_decoder = torch.load('decoderOUR_epoch_19999.pth')
        encoder.load_state_dict(saved_encoder)
        decoder.load_state_dict(saved_decoder)
        
        infere(encoder, decoder, load_data)
        
        
        #torch.save(encoder.state_dict(), 'encoderOUR_epoch_%d.pth' % (it))
        #torch.save(decoder.state_dict(), 'decoderOUR_epoch_%d.pth' % (it))
        

    


