import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.utils import to_categorical

# Generate synthetic data
num_samples = 1000
num_classes = 10
input_dim = 32

# Generate random input data for three models
X1 = np.random.rand(num_samples, input_dim)
X2 = np.random.rand(num_samples, input_dim)
X3 = np.random.rand(num_samples, input_dim)

# Generate random labels and one-hot encode them
Y = np.random.randint(num_classes, size=num_samples)
Y_one_hot = to_categorical(Y, num_classes=num_classes)

# Define the first sequential model
model1 = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu')
])

# Define the second sequential model
model2 = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu')
])

# Define the third sequential model
model3 = Sequential([
    Dense(32, activation='relu', input_shape=(input_dim,)),
    Dense(16, activation='relu')
])

# Create input layers for each model
input1 = Input(shape=(input_dim,))
input2 = Input(shape=(input_dim,))
input3 = Input(shape=(input_dim,))

# Get the outputs from each model
output1 = model1(input1)
output2 = model2(input2)
output3 = model3(input3)

# Concatenate the outputs
concat_output = Concatenate()([output1, output2, output3])

# Add more layers on top of the concatenated output
x = Dense(64, activation='relu')(concat_output)
x = Dense(32, activation='relu')(x)
final_output = Dense(num_classes, activation='softmax')(x)

# Create the final model
final_model = Model(inputs=[input1, input2, input3], outputs=final_output)

# Compile the model
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
final_model.summary()

# Train the model
final_model.fit([X1, X2, X3], Y_one_hot, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = final_model.evaluate([X1, X2, X3], Y_one_hot)
print(f"Loss: {loss}, Accuracy: {accuracy}")
