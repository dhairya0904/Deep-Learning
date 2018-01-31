
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
import sys

from createModel import CreateModel





# In[107]:

def decode_sequence(input_seq,encoder_model,decoder_model,obj):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, obj.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 1] = 1
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = obj.target_label_encoder.inverse_transform( [sampled_token_index][0] )
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > obj.max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, obj.num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# In[108]:

if __name__ == "__main__":
    
    sentence = sys.argv[1:]
    print sentence
    
    f = open('data_features.pkl', 'rb')
    data_features = pickle.load(f)
    lang_translate = CreateModel(data_features=data_features)

    model,encoder_model, decoder_model = lang_translate.getModel()
    model.load_weights('model.h5')

    encoded_sentence = lang_translate.input_label_encoder.transform(sentence)
    encoded_sentence = pad_sequences([encoded_sentence], padding='post', maxlen = lang_translate.max_encoder_seq_length)
    encoded_sentence = to_categorical(encoded_sentence, num_classes=lang_translate.num_encoder_tokens) 
    encoded_sentence = encoded_sentence.reshape((1,encoded_sentence.shape[0],encoded_sentence.shape[1]))

    print decode_sequence(encoded_sentence, encoder_model, decoder_model, lang_translate)


# In[ ]:



