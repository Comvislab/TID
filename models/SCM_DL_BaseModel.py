import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, BatchNormalization
import matplotlib.pyplot as plt
import random
import numpy as np

##Each 3 parallel model is constructed with;
#      1 input, 2 hidden layers and 1 output

class SCM_DL_Base:
    def __init__(self, 
        layer_units, 
        activations, 
        input_dim, 
        output_dim, 
        Optmzr='sgd', 
        Lossf='sparse_categorical_crossentropy',
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.01,
        deepname='SCM_DL_Base'):
        self.layer_units = layer_units
        self.activations = activations
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.Optmzr      = Optmzr
        self.Lossf       = Lossf
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.validation_split=validation_split
        self.learning_rate = learning_rate
        self.deepname='SCM_DL_Base'
        self.model = None
    
    def build_model(self):
        models=[]
        Inputs=[]
        Outputs=[]
        for modelnum in range(len(self.layer_units)):
        # Define each sequential model
            if modelnum==len(self.layer_units)-1: #Last model input size differs
                size_crop=self.input_dim - ((len(self.layer_units)-1)*(self.input_dim//3))
            else:
                size_crop=self.input_dim//3
            print("Modelnum:",modelnum," Size crop:", size_crop)
            tempmodel = Sequential([
            Dense(self.layer_units[modelnum][0], activation=self.activations[modelnum][0], input_shape=(size_crop,)),
            #BatchNormalization(),
            Dropout(0.3), # * modelnum),
            Dense(self.layer_units[modelnum][1], activation=self.activations[modelnum][1]),
            #BatchNormalization(),
            Dropout(0.3), # * modelnum),
            Dense(self.layer_units[modelnum][2], activation=self.activations[modelnum][2]),
            #BatchNormalization(),
            Dropout(0.3), #* modelnum),
            ])

        # Create input layers and outputs for each model
            tempInput     = Input(shape=(size_crop,))
            Inputs.append(tempInput)
            Outputs.append(tempmodel(tempInput))

        # Concatenate all the outputs
        # concat_output = Concatenate()(Outputs)
  
        concat_0_1 = Concatenate()([Outputs[0], Outputs[1]])  
        concat_0_2 = Concatenate()([Outputs[0], Outputs[2]])  
        concat_1_2 = Concatenate()([Outputs[1], Outputs[2]])  
               
  
        #  Add more layers on top of the concatenated output
        x1 = Dense(128, activation='sigmoid')(concat_0_1)
        x2 = Dense(128, activation='sigmoid')(concat_0_2)
        x3 = Dense(128, activation='sigmoid')(concat_1_2)

        concat_output = Concatenate()([x1,x2,x3])

        x = Dense(128, activation='relu')(concat_output)
       
        # Assuming you have number of classes = output_dim
        # x = concat_output
        final_output = Dense(self.output_dim, activation='softmax')(x)

        # Create the final model
        self.model = Model(inputs=Inputs, outputs=final_output)

    def compile_model(self, optimizer, learning_rate, loss, metrics=['accuracy']):
        optimizer = SGD(learning_rate=learning_rate) if optimizer.lower() == 'sgd' else optimizer
        loss = SparseCategoricalCrossentropy() if loss.lower() == 'sparse_categorical_crossentropy' else loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # Compile the model (shortcut)
        #self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, epochs, batch_size, validation_split):
        size_crop = self.input_dim//3
        X_tr1 = np.array([row[:size_crop].tolist() for row in X_train])
        X_tr2 = np.array([row[size_crop:2*size_crop].tolist() for row in X_train])
        X_tr3 = np.array([row[2*size_crop:].tolist() for row in X_train])
         
        #input ("DIKKAT")
        #print(X_train, type(X_train))
        #input ("DIKKAT2")
        #print(X_tr1, type(X_tr1))
        #input("BEKLE ve YORUMLA")
        print("Size of X_train", np.shape(X_train))
        print("Size of X_tr1", np.shape(X_tr1))
        print("Size of X_tr2", np.shape(X_tr2))
        print("Size of X_tr3", np.shape(X_tr3))
        
        #X_tr2 = np.roll(X_tr1, self.input_dim//3, axis=1)
        history = self.model.fit([X_tr1, X_tr2, X_tr3], y_train, epochs=epochs, batch_size=batch_size) #, validation_split=validation_split)
        return history
       
    def evaluate(self, X_test, y_test):
        size_crop = self.input_dim//3
        X_tt1 = np.array([row[:size_crop].tolist() for row in X_test])
        X_tt2 = np.array([row[size_crop:2*size_crop].tolist() for row in X_test])
        X_tt3 = np.array([row[2*size_crop:].tolist() for row in X_test])
        
        #X_tt3 = np.roll(X_tt2, self.input_dim//3, axis=1)

        return self.model.evaluate([X_tt1, X_tt2, X_tt3], y_test)

    def predict(self, X_new_data):
        size_crop = self.input_dim//3
        X_tn1 = np.array([row[:size_crop].tolist() for row in X_new_data])
        X_tn2 = np.array([row[size_crop:2*size_crop].tolist() for row in X_new_data])
        X_tn3 = np.array([row[2*size_crop:].tolist() for row in X_new_data])

        #X_tn3 = np.roll(X_tn2, self.input_dim//3, axis=1)

        return self.model.predict([X_tn1,X_tn2,X_tn3])



class ShallowDL:
    def __init__(self, 
        layer_units, 
        activations, 
        input_dim, 
        output_dim, 
        Optmzr='sgd', 
        Lossf='sparse_categorical_crossentropy',
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.01,
        deepname='ShallowDL'):
        self.layer_units = layer_units
        self.activations = activations
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.Optmzr      = Optmzr
        self.Lossf       = Lossf
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.validation_split=validation_split
        self.learning_rate = learning_rate
        self.deepname='ShallowDL'
        self.model = None
    
    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_units)):
            if i == 0:
                model.add(Dense(units=self.layer_units[i], activation=self.activations[i], input_dim=self.input_dim))
            else:
                model.add(Dense(units=self.layer_units[i], activation=self.activations[i]))
                model.add(Dropout(0.3))
                #model.add(BatchNormalization()),
        model.add(Dense(units=self.output_dim, activation='softmax'))  # Output layer
        self.model = model
    
    def compile_model(self, optimizer, learning_rate, loss, metrics=['accuracy']):
        optimizer = SGD(learning_rate=learning_rate) if optimizer.lower() == 'sgd' else optimizer
        loss = SparseCategoricalCrossentropy() if loss.lower() == 'sparse_categorical_crossentropy' else loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, X_train, y_train, epochs, batch_size, validation_split):
        print("Size of X_train", np.shape(X_train))
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X_new_data):
        return self.model.predict(X_new_data)
        
    def summary(self):
        self.model.summary()

        