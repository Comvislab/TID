# DeepLearningModels File Modification Date: 30.07.2024 - 14:20

# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, BatchNormalization
import matplotlib.pyplot as plt
import random
import numpy as np
# Optmz = 'sgd'
# Lossf = 'sparse_categorical_crossentropy'
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
        
        
        
        

"""
# Define the parameters for the neural network
layer_units = [64, 32]  # Number of units for each layer (excluding input and output layers)
activations = ['relu', 'relu']  # Activation functions for each layer
input_dim = 784  # Input dimension (e.g., for MNIST, flattened image size)
output_dim = 10  # Output dimension (e.g., number of classes)

# Initialize the custom neural network
custom_nn = CustomNeuralNetwork(layer_units, activations, input_dim, output_dim)

# Build the model
custom_nn.build_model()

# Compile the model
custom_nn.compile_model()

# Train the model (Assuming you have X_train and y_train ready)
history = custom_nn.train(X_train, y_train)

# Evaluate the model (Assuming you have X_test and y_test ready)
loss, accuracy = custom_nn.evaluate(X_test, y_test)

# Make predictions
predictions = custom_nn.predict(X_new_data)
"""

