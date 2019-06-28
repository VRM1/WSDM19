from keras.layers import Dense, Input, Lambda, Embedding, Flatten
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np
from tqdm import tqdm
from keras.backend import int_shape
from keras import metrics
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from collections import defaultdict
from keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

# this is the class for creating the DSAE with embeding layer
class DSAEB:
    
    def __init__(self,name,inter_dim, z_dim, inpt_dim,bath_size,typ):
        
        self.inpt_dim = inpt_dim
        self.inter_dim = inter_dim
        self.z_dim = z_dim
        self.bath_size = bath_size
        self.epochs = 50
        self.name = name
        self.typ = typ

    def _dsae_loss(self,z_mean,z_vari):
        def loss(x, x_decoded_mean):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            # E[log P(X^{a}|z)]
            recon_a = self.inpt_dim * metrics.mean_squared_error(x, x_decoded_mean)
            # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl_a = - 0.5 * K.sum(1 + z_vari - K.square(z_mean) - K.exp(z_vari), axis=-1)            
            fin_loss = recon_a + kl_a
            return fin_loss
        return loss
    
    def VAE(self,num):
    
        # the number of nodes for the intermediate dimension
        epsilon_std = 1.0
        # mean and variance for isotropic normal distribution

        # create a place holder for input layer
        x = Input(shape=(self.inpt_dim,),name='InputX-'+num)
        g = Dense(self.inter_dim,activation='relu')(x)
        z_mean = Dense(self.z_dim)(g)
        z_vari = Dense(self.z_dim)(g)
        
        # create the latent layer
        z = Lambda(self._SampleZ,output_shape=(self.z_dim,), arguments={'epsilon_std':epsilon_std, 'batch_size':self.bath_size,
                                                              'latent_dim':self.z_dim})([z_mean,z_vari])
        # decoder, intermediate layer
        decoded_g = Dense(self.inter_dim,activation='relu')
        
        decoded_mean = Dense(self.inpt_dim,name='ReconstX-'+num)
        g_decoded = decoded_g(z)
        x_decoded_mean = decoded_mean(g_decoded)
        # create the complete vae model
        vae = Model(x,x_decoded_mean)
        # create just the encoder
        encoder = Model(x, z_mean)

        # generator, from latent space to reconstructed inputs
        decoder_input = Input(shape=(self.z_dim,),name='InputZ-'+num)
        _g_decoded = decoded_g(decoder_input)
        _x_decoded_mean = decoded_mean(_g_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        return (vae,x,z_mean,z_vari,z,x_decoded_mean,encoder,generator)


    def _SampleZ(self, args,epsilon_std,latent_dim,batch_size):
        
        z_mean, z_var = args
        # sample Z from isotropic normal
        epslon = K.random_normal(shape=(1, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + z_var * epslon


    # method of the classifier neural network
    def _ClassifierNN(self,z_a,z_b,_z_a,_z_b):
        
        if self.typ == 'binary':
            lbl = 2
        else:
            lbl = 4
        # concatenate the two latent layers from VAE_a and VAE_b
        conc = concatenate([z_a, z_b])
        # concatenation layer for the generator
        _conc = concatenate([_z_a,_z_b])
        # intermediate layer
        interim = Dense(self.inter_dim,activation='relu',name='Intermed-Classify')
        # the final predictor layer
        predictor = Dense(lbl,activation='softmax',name='Pred-SubsNCompl')

        # create the training model for the full end-to-end
        g_za_zb = interim(conc)
        # we have four labels 0,1 are for substitutes and 2,3 are for compliments 
        y_label = predictor(g_za_zb)

        # create a testing/generator model
        _g_za_zb = interim(_conc)
        # we have four labels 0,1 are for substitutes and 2,3 are for compliments
        _y_label = predictor(_g_za_zb)

        return (y_label,_y_label)
    
    # method that get the DSAE model
    def getModels(self):
        
        # create the first VAE model (i.e., vae_a)
        vae_a,x_a,z_mean_a,z_vari_a,z_a,x_decoded_mean,encoder,generator_a = self.VAE('A')
        #     create the second VAE model (i.e., vae_b)
        vae_b,x_b,z_mean_b,z_vari_b,z_b,x_decoded_mean,encoder,generator_b = self.VAE('B')
        # create the classifier neural network for both end-to-end model and the generator model
        classifier_nn,classifier_gen = self._ClassifierNN(z_a,z_b,generator_a.inputs[0],generator_b.inputs[0])
        # create the complete model
        LVAE = Model(inputs=[vae_a.inputs[0],vae_b.inputs[0]],\
                     outputs=[vae_a.outputs[0],vae_b.outputs[0],classifier_nn])
        
        #     Compile the autoencoder computation graph
        LinkPredictor = Model(inputs=[LVAE.inputs[0],LVAE.inputs[1]],outputs=[LVAE.outputs[2]])
        # the generator model
        LinkGenerator = Model(inputs=[generator_a.inputs[0], generator_b.inputs[0]],\
                              outputs=[generator_a.outputs[0],generator_b.outputs[0],classifier_gen])
        # model for evaluation
        LVAE.compile(optimizer="adam", loss=[self._dsae_loss(z_mean_a,z_vari_a), \
                                             self._dsae_loss(z_mean_b,z_vari_b),\
                                             'binary_crossentropy'], loss_weights=[0.3,0.3,0.9])

        cp = [ModelCheckpoint(filepath='output/DSAE_'+self.name+'.hdf5', verbose=1, monitor='val_loss', mode='min',\
                              save_best_only=True)]
        LinkPredictor.compile(optimizer="adam", loss='binary_crossentropy',metrics=['accuracy'])
        # plot_model(LVAE,to_file='LVAE-end-to-end.png',show_shapes=True)
        # plot_model(LinkPredictor,to_file='LinkPredictor.png',show_shapes=True)
        # plot_model(LinkGenerator,to_file='LinkGenerator.png',show_shapes=True)
        return (LVAE,LinkPredictor,cp,generator_a)
    
    def TestVAE(self):
        
        # create the first VAE model (i.e., vae_a)
        vae_a,x_a,z_mean_a,z_vari_a,z_a,x_decoded_mean,encoder = self.VAE()
        vae_a.compile(optimizer='adam', loss=self._dsae_loss(z_mean_a,z_vari_a))
        plot_model(vae_a,to_file='VAE.png',show_shapes=True)
        return vae_a
