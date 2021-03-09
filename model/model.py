import itertools
import torch
import torch.nn as nn

import os
os.chdir( path = "C:/Users/msi/Desktop/work/study/USAD/model" )
from autoencoder import Encoder, Decoder

class USAD_model( nn.Module ):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoder = Encoder( input_size, latent_size )
        self.decoder_1 = Decoder( latent_size, input_size )
        self.decoder_2 = Decoder( latent_size, input_size )
    
    def forward(self, batch, n):
        z = self.encoder( batch )
        w1 = self.decoder_1( z )
        w2 = self.decoder_2( z )
        w3 = self.decoder_2( self.encoder( w1 ) )
        
        loss_1 = (1/n)*torch.mean((batch-w1)**2) + (1-(1/n))*torch.mean((batch-w3)**2)
        loss_2 = (1/n)*torch.mean((batch-w2)**2) - (1-(1/n))*torch.mean((batch-w3)**2)
        
        return(loss_1, loss_2)
    
    def predict(model, test_loader, device, alpha = .5, beta = .5):
        score_list = []
        for [batch] in test_loader:
            batch = batch.to(device)
            w1 = model.decoder_1( model.encoder(batch) )
            w2 = model.decoder_2( model.encoder(batch) )
            score = alpha * torch.mean((batch-w1)**2, axis = 1) + \
                beta * torch.mean((batch-w2)**2, axis = 1)
            score_list.append( score )
        score_list = list(itertools.chain(*score_list))
        return( score_list )