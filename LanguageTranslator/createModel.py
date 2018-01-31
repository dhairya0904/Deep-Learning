
# coding: utf-8

# In[35]:

import numpy as np
import pandas as pd
import nltk
seed = 7
np.random.seed(seed)

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM

import pickle


# In[36]:

class CreateModel(object):
    
    def __init__(self, path=None, epochs = 10, train_len = 1000, batch_size = 64, latent_dim = 256, data_features = None):

    	self.path = path
        self.epochs = epochs
        self.train_len = train_len
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
        self.target_label_encoder = None 
        self.max_encoder_seq_length = None 
        self.max_decoder_seq_length = None
        self.data = None
        self.input = None
        self.target = None
        # for training
        if path is not None:
            self.data = pd.read_csv(path,delimiter="\t",header=None, encoding = 'utf8')[:self.train_len]
            self.input = self.data[0]
            self.target = self.data[1]
            self.input = self.input.apply(lambda x: x.lower())
            self.target = self.target.apply(lambda x: x.lower())

        # for decoding
        else:
        	self.num_encoder_tokens = data_features[0]
        	self.num_decoder_tokens = data_features[1]
        	self.target_label_encoder = LabelEncoder()
        	self.target_label_encoder.classes_ = data_features[2]
        	self.input_label_encoder = LabelEncoder()
        	self.input_label_encoder.classes_ = data_features[3]
        	self.max_encoder_seq_length = data_features[4] 
        	self.max_decoder_seq_length = data_features[5]
    
    
    def preprocess(self):
        
        #tokenize data
        
        self.input_token = self.input.apply( lambda x: list(x) )
        self.target_token = self.target.apply( lambda x: list(x) )
        
        # add '\n' as sentinel to end of sentence
        
        self.input_token = self.input_token.apply( lambda x: x+['\n'] )
        self.target_token = self.target_token.apply( lambda x: x+['\n'] ) 
        
        #create input and target corpus
        
        input_corpus = [] 
        self.input_token.apply( lambda x: input_corpus.extend( x ) )
        self.input_corpus = list(set(input_corpus) )
        
        target_corpus = [] 
        self.target_token.apply( lambda x: target_corpus.extend( x ) )
        self.target_corpus = list(set(target_corpus) )
        
        self.num_encoder_tokens = len(self.input_corpus)
        self.num_decoder_tokens = len(self.target_corpus)
        
        #sequence length
        
        self.max_encoder_seq_length = max(self.input_token.apply(lambda x: len(x)) )
        self.max_decoder_seq_length = max(self.target_token.apply(lambda x: len(x)) )
        
        #Label Encoder
        
        self.input_label_encoder = LabelEncoder().fit(self.input_corpus)
        self.target_label_encoder = LabelEncoder().fit(self.target_corpus)
        
        encoder_input = self.input_token.apply( lambda x: self.input_label_encoder.transform(x) )
        decoder_input = self.target_token.apply( lambda x: self.target_label_encoder.transform(x) )
        decoder_target = decoder_input.apply(lambda x: x[1:])
        
        #padding the sequence
        
        encoder_input = pad_sequences(encoder_input, padding='post', maxlen = self.max_encoder_seq_length)
        decoder_input = pad_sequences(decoder_input, padding='post', maxlen = self.max_decoder_seq_length)
        decoder_target = pad_sequences(decoder_target, padding='post', maxlen = self.max_decoder_seq_length)
        
        
        #one hot encoding
        
        X_hot = [to_categorical(x, num_classes=self.num_encoder_tokens) for x in encoder_input ]
        encoder_input_data = np.dstack( X_hot)
        self.encoder_input_data = np.rollaxis(encoder_input_data,-1)

        y_hot = [to_categorical(x, num_classes=self.num_decoder_tokens) for x in decoder_input ]
        decoder_input_data = np.dstack( y_hot)
        self.decoder_input_data = np.rollaxis(decoder_input_data,-1)

        target_hot = [to_categorical(x, num_classes=self.num_decoder_tokens) for x in decoder_target ]
        decoder_target_data = np.dstack( target_hot)
        self.decoder_target_data = np.rollaxis(decoder_target_data,-1)
        
    def getModel(self):
        
        #hyperparameters
        epochs = self.epochs
        batch_size = self.batch_size
        latent_dim = self.latent_dim
        
        
        #set up the encoder

        encoder_inputs = Input( shape = (None, self.num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state = 1)
        
        encoder_outputs,state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h,state_c]
        
        #set up the decoder
        
        decoder_inputs = Input( shape = (None, self.num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_state= True, return_sequences = True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        #training model
        self.train_model = Model( [encoder_inputs, decoder_inputs], decoder_outputs)
        
        #encoder model
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        #decoder model
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        
        return self.train_model, self.encoder_model, self.decoder_model
    
    def fit(self):
        self.preprocess()
        model,_,_  = self.getModel()
    
        model.compile( optimizer = 'rmsprop', loss='categorical_crossentropy' ) 
        model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size= self.batch_size,
                  epochs= self.epochs,
                  validation_split=0.2)   
        model.save("model.h5")
        
        data_features = [self.num_encoder_tokens, self.num_decoder_tokens, 
                        self.target_label_encoder.classes_, 
                        self.input_label_encoder.classes_,
                        self.max_encoder_seq_length,
                        self.max_decoder_seq_length]
        
        with open("data_features.pkl",'wb') as f:
            pickle.dump(data_features, f)
            
        return model


if __name__ == "__main__":
	create = CreateModel('fra.txt',epochs = 100, train_len = 5000)
	create.fit()