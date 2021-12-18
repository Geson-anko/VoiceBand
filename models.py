import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchsummaryX import summary
from torch.nn.utils import weight_norm, remove_weight_norm
from utils import get_padding, get_conv1d_outlen, init_weights, get_padding_down, get_padding_up,walk_ratent_space
from typing import Tuple
from torchsummaryX import summary
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


LRELU_SLOPE = 0.1
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        assert len(dilation) == 3
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Encoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        rks = h.resblock_kernel_sizes
        rds = h.resblock_dilation_sizes
        drs = h.downsample_rates
        drks = h.downsample_kernel_sizes
        dci = h.downsample_initial_channel
        
        self.num_kernels = len(rks)
        self.num_downsamples = len(drs)
        self.conv_pre = weight_norm(nn.Conv1d(1, dci, 7,1,3))
        
        # get expected input lengthes and output lengths
        init_len = h.n_fft
        self.L_ins = [init_len]
        self.L_outs = []
        for r in drs:
            lo = int(init_len/r)
            self.L_outs.append(lo)
            self.L_ins.append(lo)
            init_len = lo
        self.L_outs.append(1)

        # get downsampling paddings
        self.pads = []
        for i,r in enumerate(drs):
            pad = get_padding_down(self.L_ins[i],self.L_outs[i],drks[i],r)
            self.pads.append(pad)
    
        # get downsampling channels
        self.channels = []
        for i in range(len(drs)+1):
            self.channels.append(dci*(2**i))
    
        self.dns = nn.ModuleList()
        for i, (u, k) in enumerate(zip(drs, drks)):
            self.dns.append(weight_norm(
                nn.Conv1d(self.channels[i], self.channels[i+1],k,u,self.pads[i])
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.dns)):
            ch = self.channels[i+1]
            for j,(k,d) in enumerate(zip(rks,rds)):
                self.resblocks.append(ResBlock(ch,k,d))
        
        self.conv_post = weight_norm(nn.Conv1d(self.channels[-1],h.ratent_dim,self.L_ins[-1]))
        self.conv_post_var = weight_norm(nn.Conv1d(self.channels[-1],h.ratent_dim,self.L_ins[-1]))
        self.dns.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_post_var.apply(init_weights)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.dns[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        mean = self.conv_post(x)
        var = F.softplus(self.conv_post_var(x)) + 1e-8
        
        return mean,var

    def dual_flow(self, x1:torch.Tensor, x2:torch.Tensor,with_random:bool=True) -> torch.Tensor:
        mean1,var1 = self.forward(x1)
        mean2,var2 = self.forward(x2)
        if with_random:
            out1 = self.random_sample(mean1,var1)
            out2 = self.random_sample(mean2,var2)
        else:
            out1,out2 = mean1,mean2
        out = torch.cat([out1, out2], dim=1) #.tanh() # notanh
        return out

    @staticmethod
    def random_sample(mean:torch.Tensor, var:torch.Tensor):
        return mean + torch.randn_like(mean)*torch.sqrt(var)

    def summary(self):
        dummy = torch.randn(1,1,self.h.n_fft)
        summary(self, dummy)

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.dns:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class Decoder(nn.Module):
    def __init__(self, h) -> None:
        super().__init__()
        self.h = h
        rks = h.resblock_kernel_sizes
        rds = h.resblock_dilation_sizes
        uik = h.upsample_initial_kernel
        urs = h.upsample_rates
        urks = h.upsample_kernel_sizes
        uic = h.upsample_initial_channel
        self.out_len = h.n_fft +h.hop_len

        self.num_kernels = len(rks)
        self.num_upsamples = len(urs)
        self.conv_pre = weight_norm(nn.ConvTranspose1d(h.ratent_dim*2, uic,uik))

        # get expected input lengthes and output lengthes
        init_len = uik
        self.L_ins = [init_len]
        self.L_outs = []
        for r in urs:
            lo = init_len * r
            self.L_ins.append(lo)
            self.L_outs.append(lo)
            init_len = lo
        
        # get upsampling paddings
        self.pads = []
        for i,r in enumerate(urs):
            pad = get_padding_up(self.L_ins[i],self.L_outs[i],urks[i],r)
            self.pads.append(pad)
        
        # get upsampling channels
        self.channels = [uic]
        ch = uic
        for i in range(len(urs)):
            self.channels.append(int(ch/(2**i)))

        self.ups = nn.ModuleList()
        for i, (u,k) in enumerate(zip(urs,urks)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(self.channels[i], self.channels[i+1],k,u,self.pads[i])
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.channels[i+1]
            for j,(k,d) in enumerate(zip(rks,rds)):
                self.resblocks.append(ResBlock(ch,k,d))
        self.conv_post = weight_norm(nn.Conv1d(self.channels[-1],1,7,1,3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        l = x.size(-1)
        start = int((l - self.out_len)/2)
        x = x[:,:,start:start+self.out_len]
        x = torch.tanh(x)
        return x

    def summary(self):
        dummy = torch.randn(1,self.h.ratent_dim*2,1)
        summary(self,dummy)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class VoiceBand(pl.LightningModule):
    def __init__(self, h,dtype:torch.dtype=torch.float,device:torch.device='cpu') -> None:
        super().__init__()
        self.h = h
        self.reset_seed()
        self.encoder = Encoder(h).type(dtype).to(self.device)
        self.decoder = Decoder(h).type(dtype).to(self.device)
        self.n_fft = h.n_fft
        self.ratent_dim = h.ratent_dim
        self.walking_steps = int(h.breath_len / h.hop_len) + 1
        self.walking_resolution = h.walking_resolution
        self.out_len = self.decoder.out_len
        self.view_interval = 10
        self.kl_lambda = h.kl_lambda
        # training settings
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()

        self.actions = walk_ratent_space(self.ratent_dim, self.walking_steps,self.walking_resolution,device=device,dtype=dtype)
        

    def forward(self, x1:torch.Tensor,x2:torch.Tensor) -> torch.Tensor:
        """
        x1: (-1, 1, n_fft)
        x2: (-1, 1, n_fft)
        """ 
        mean1,var1 = self.encoder.forward(x1)
        mean2,var2 = self.encoder.forward(x2)
        mean,var = torch.cat([mean1,mean2],dim=1),torch.cat([var1,var2],dim=1)
        out = self.encoder.random_sample(mean,var)#.tanh()# notanh
        out = self.decoder(out)
        return out,mean,var

    def on_fit_start(self) -> None:
        self.logger.log_hyperparams(self.h)

    def training_step(self, batch:Tuple[torch.Tensor], batch_idx) -> torch.Tensor:
        """
        batch : (-1, ch, n_fft+hop_len)
        """
        sound, = batch
        sound = sound.type(self.dtype)
        if self.h.random_gain:
            sound= self.random_gain(sound)

        x1,x2,ans = sound[:,:,:self.h.n_fft], sound[:,:,-self.h.n_fft:], sound 
        out,mean,var = self.forward(x1,x2)
        
        mse = self.MSE(ans, out)
        mae = self.MAE(ans,out)
        KL = 0.5*torch.sum(
            torch.pow(mean,2) +
            var -
            torch.log(var) -1 
        ).sum() / out.size(0)
        marginal_likelihood = F.binary_cross_entropy(0.5*out+1,ans,reduction="sum")/out.size(0)
        loss = marginal_likelihood + KL
        #loss = self.kl_lambda * KL + mse
        self.log("loss",loss)
        self.log("mse",mse)
        self.log("mae",mae)
        self.log("KL div",KL)
        self.log("Marginal likelihood",marginal_likelihood)
        return loss

    @torch.no_grad()
    def on_epoch_end(self) -> None:
        """
        walk through the ratent space and log audio wave.
        """
        if self.current_epoch%self.view_interval !=0:
            return

        self.actions = walk_ratent_space(self.ratent_dim, self.walking_steps,self.walking_resolution,
                                        device=self.device,dtype=self.dtype)
        wave = None
        for act in self.actions.unsqueeze(1):
            wave= self.predict_one_step(act,wave)
        wave = wave.squeeze(0).T.detach().cpu().numpy()

        # tensorboard logging
        tb:SummaryWriter = self.logger.experiment
        tb.add_audio("Ratent space audio",wave, self.current_epoch,self.h.frame_rate)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(wave)
        tb.add_figure("Walked wave",fig, self.current_epoch)
        return 

    def random_gain(self, sound:torch.Tensor) -> torch.Tensor:
        n,c,l = sound.shape
        maxes= sound.view(n,c*l).abs().max(dim=1,keepdim=True).values.unsqueeze(-1)
        maxes[maxes==0.0] = 1.0
        gains = torch.rand_like(maxes)
        sound = (sound/maxes) * gains
        return sound

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), self.h.lr,[self.h.adam_b1,self.h.adam_b2])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.h.lr_decay)
        scheduler.last_epoch=self.trainer.max_epochs
        return [optim],[scheduler]

    silence = None
    def set_silence(self):
        self.silence = torch.zeros(1,self.h.sample_ch,self.n_fft,device=self.device,dtype=self.dtype)
    def set_view_interval(self, interval:int=None):
        if interval:
            self.view_interval= interval
            
    def predict_one_step(self, action:torch.Tensor,previous_wave:torch.Tensor=None) -> torch.Tensor:
        """
        action : (-1, ratent_dim, 1)
        previous_wave : (-1,ch, l)
        """
        if previous_wave is None:
            if self.silence is None:
                self.set_silence()
            previous_wave = self.silence

        assert len(action.shape) == 3
        assert len(previous_wave.shape) == 3

        if previous_wave.size(-1) < self.n_fft :
            pad_len = self.n_fft - previous_wave.size(-1)
            n,c,l = previous_wave.shape
            pad = torch.zeros(n,c,pad_len,dtype=previous_wave.dtype,device=previous_wave.device)
            previous_wave = torch.cat([pad,previous_wave],dim=-1)
        
        enc_in = previous_wave[:,:,-self.n_fft:].to(self.dtype).to(self.device)
        encoded = self.encoder.forward(enc_in)[0]#.tanh()# notanh
        dec_in = torch.cat([encoded,action],dim=1)
        d_out = self.decoder.forward(dec_in)[:,:,self.n_fft:].type_as(previous_wave)
        wave = torch.cat([previous_wave,d_out],dim=-1)
        return wave
        
    def reset_seed(self):
        seed = self.h.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    def summary(self,tensorboard:bool = True):
        dummy = torch.randn(1,1,self.n_fft)
        summary(self, dummy,dummy)
        if tensorboard:
            writer = SummaryWriter()
            writer.add_graph(self, [dummy,dummy])

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()
        self.decoder.remove_weight_norm()




if __name__ == '__main__':
    from utils import load_config
    config = load_config("hparams/origin.json")
    model = VoiceBand(config)
    model.summary()
    model.remove_weight_norm()