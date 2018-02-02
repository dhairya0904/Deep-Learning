import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

seed= 7

np.random.seed(7)

import os


# In[88]:

class Model(object):
    
    def __init__(self, target_size = (150, 150, 3)):
        self.vgg16 = VGG16(include_top=False,input_shape= target_size )
        self.target_size = target_size
        
    def generateData(self, data_path):
        datagen = ImageDataGenerator(rescale=1./255)
        target_size = self.target_size
        batch_size = 16
        
        generator = datagen.flow_from_directory(data_path,
                                             batch_size= batch_size,
                                             shuffle = False,
                                             target_size = target_size[:-1],
                                             class_mode = "categorical")
        self.classes = len(generator.class_indices)
        
        
        i = 0
        nImages = len(generator.filenames)
        out_shape = (nImages,) + tuple(self.vgg16.layers[-1].output_shape)[1:]

        features = np.zeros(shape= out_shape )
        labels = np.zeros(shape=(nImages,self.classes))

        for inputs_batch, labels_batch in generator:
            features_batch = self.vgg16.predict(inputs_batch)
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= nImages:
                break
                
        self.flattened_len = reduce(lambda x, y: x*y,list(self.vgg16.layers[-1].output_shape)[1:] )
        
        labels = labels
        features = np.reshape(features, (nImages, self.flattened_len))
    
        return features, labels, generator
    
    def fit(self, train_features, train_labels, validation_features, validation_labels,
           epochs = 20, batch_size = 16, verbose = 0):
        
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim = self.flattened_len))
        model.add( Dropout(0.4) )
        
        model.add( Dense(self.classes, activation='softmax'))
        
        model.compile( optimizer = 'rmsprop', 
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
        
        model.fit(train_features, train_labels,
                  batch_size = batch_size,
                  epochs = epochs,
                  validation_data = (validation_features, validation_labels),
                  verbose = verbose
                 )
        self.model = model
        return self.model 
    
    def predict(self, features):
        return self.model.predict(features)
    
    def predict_class(self, features):
        return self.model.predict_classes(features)
        


