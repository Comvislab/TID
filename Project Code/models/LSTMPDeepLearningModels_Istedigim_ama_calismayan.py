import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, BatchNormalization, LSTM
import matplotlib.pyplot as plt
import random

##Each 3 parallel model is constructed with;
#      1 input, 2 hidden layers and 1 output

class LSTMParShallowDL:
    def __init__(self, 
        layer_units, 
        activations, 
        num_timesteps,
        input_dim, 
        output_dim, 
        Optmzr='sgd', 
        Lossf='sparse_categorical_crossentropy',
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.01,
        deepname='LSTMParShallowDL'):
        self.layer_units = layer_units
        self.activations = activations
        self.num_timesteps=num_timesteps
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.Optmzr      = Optmzr
        self.Lossf       = Lossf
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.validation_split=validation_split
        self.learning_rate = learning_rate
        self.deepname='LSTMParShallowDL'
        self.model = None
    
    def build_model(self):
        models=[]
        Inputs=[]
        Outputs=[]
        for modelnum in range(len(self.layer_units)):
        # Define each sequential model
            tempmodel = Sequential([
            LSTM(self.layer_units[modelnum][0], return_sequences=True, input_shape=(self.num_timesteps,self.input_dim),recurrent_dropout=0.5),
            BatchNormalization(),
            Dropout(0.1),
            LSTM(self.layer_units[modelnum][1], recurrent_dropout=0.5),
            BatchNormalization(),
            Dropout(0.1),
            LSTM(self.layer_units[modelnum][2], recurrent_dropout=0.5),
            BatchNormalization(),
            Dropout(0.1),
            ])

        # Create input layers and outputs for each model
            tempInput     = Input(shape=(self.num_timesteps, self.input_dim,))
            Inputs.append(tempInput)
            Outputs.append(tempmodel(tempInput))


        # Concatenate the outputs
        concat_output = Concatenate()(Outputs)
        
        #  Add more layers on top of the concatenated output
        x = Dense(256, activation='relu')(concat_output)
        #x = Dense(128, activation='relu')(x)
       
        # Assuming you have number of classes = output_dim
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
        history = self.model.fit([X_train, X_train, X_train], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history
       
    def evaluate(self, X_test, y_test):
        return self.model.evaluate([X_test, X_test, X_test], y_test)

    def predict(self, X_new_data):
        return self.model.predict([X_new_data,X_new_data,X_new_data])



    # Create rolling window sequences
    def create_rolling_window_sequences(data, num_timesteps):
        sequences = []
        for i in range(len(data) - num_timesteps + 1):
            sequences.append(data[i:i + num_timesteps])
        return np.array(sequences)

