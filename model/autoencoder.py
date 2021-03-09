import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size ):
        super().__init__()
        self.linear_1 = nn.Linear( in_features = input_size, out_features = int(input_size/2) )
        self.linear_2 = nn.Linear( in_features = int(input_size/2), out_features = int(input_size/4) )
        self.linear_3 = nn.Linear( in_features = int(input_size/4), out_features = latent_size )
        self.relu = nn.ReLU( inplace = True )
    def forward( self, x0 ):
        x = self.linear_1( x0 )
        x = self.relu( x )
        x = self.linear_2( x )
        x = self.relu( x )
        x = self.linear_3( x )
        x = self.relu( x )
        return( x )

class Decoder(nn.Module):
    def __init__(self, latent_size, output_size ):
        super().__init__()
        self.linear_1 = nn.Linear( in_features = latent_size, out_features = int(output_size/4) )
        self.linear_2 = nn.Linear( in_features = int(output_size/4), out_features = int(output_size/2) )
        self.linear_3 = nn.Linear( in_features = int(output_size/2), out_features = output_size )
        self.relu = nn.ReLU( inplace = True )
        self.sigmoid = nn.Sigmoid()
    def forward( self, x ):
        x = self.linear_1( x )
        x = self.relu( x )
        x = self.linear_2( x )
        x = self.relu( x )
        x = self.linear_3( x )
        x = self.sigmoid( x )
        return( x )
