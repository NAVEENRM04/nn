CODING:
Ex.1 Implement simple vector addition in TensorFlow


import tensorflow as tf
# Create two constant tensors
vector1 = tf.constant([1.0, 2.0, 3.0])
vector2 = tf.constant([4.0, 5.0, 6.0])
# Perform vector addition
result = tf.add(vector1, vector2)
# Start a TensorFlow session
with tf.Session() as sess:
# Run the session to compute the result
output = sess.run(result)
print(output)

ex 2 Implement a regression model in Keras.


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Generate some example data for regression
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)
# Define a sequential model
model = keras.Sequential()
# Add a single dense layer with one output unit (for regression)
model.add(layers.Dense(1, input_shape=(1,)))
# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')
# Train the model
model.fit(X, y, epochs=100, verbose=1)
# Make predictions
predictions = model.predict(X)
# Evaluate the model if needed
loss = model.evaluate(X, y)
print(f"Mean Squared Error: {loss}")
# Print the model's summary
model.summary()

ex3 Implement a perceptron in TensorFlow/Keras Environment.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
# Generate some example data for a logical OR operation
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
# Define a sequential model
model = keras.Sequential()
# Add a single dense layer with one output unit (perceptron)
model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
# Compile the model
model.compile(optimizer='sgd',loss='mean_squared_error', 
metrics=['accuracy'])
# Train the model
model.fit(X, y, epochs=1000, verbose=1)
# Make predictions
predictions = model.predict(X)
print(predictions)
# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Mean Squared Error: {loss}")
print(f"Accuracy: {accuracy}")

ex4 Implement a Feed-Forward Network in TensorFlow/Keras

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
# Generate some example data for a classification task
X,y= make_classification(n_samples=1000, n_features=20, random_state=42)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
model = keras.Sequential()
# Add a dense hidden layer with ReLU activation
model.add(Dense(64, input_shape=(20,), activation='relu'))
# Add an output layer with a single unit and sigmoid activation (for binary 
classification)
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy', 
metrics=['accuracy'])
# Train the model
model.fit(X_train,y_train,epochs=10,batch_size=32,verbose=1, 
validation_data=(X_test, y_test))
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")


ex5 Implement an Image Classifier using CNN in ensorFlow/Keras.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0 # Normalize pixel values
# Expand dimensions to match the expected input shape for CNN
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]
# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Create a CNN model
model = keras.Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)) 
MaxPooling2D((2, 2)),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D((2, 2)),
Conv2D(64, (3, 3), activation='relu'),
Flatten(),
Dense(64, activation='relu'),
Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

Ex:6 Improve the Deep learning model by fine tuning hyperparameter


import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch
# Define the base model
def build_model(hp):
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
# Hyperparameters to tune
hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
model.add(keras.layers.Dense(units=hp_units, activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax')) 
model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning
_rate),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
return model
# Initialize the tuner
tuner = RandomSearch(
build_model,
objective='val_accuracy',
max_trials=10,
num_initial_points=3,
directory='my_dir',
project_name='my_project')
# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
# Train the final model with the best hyperparameters
best_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


exp 7   Implement a Transfer Learning concept in Image Classification.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Step 1: Choose a pre-trained model and load it
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
include_top=False,weights='imagenet')
# Step 2: Build a custom classifier on top of the pre-trained model
model = models.Sequential([
base_model,
layers.GlobalAveragePooling2D(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(num_classes, activation='softmax') # Set num_classes to the 
number of your classes
])
# Step 3: Freeze the pre-trained layers
for layer in base_model.layers:
layer.trainable = False
# Step 4: Compile the model
model.compile(optimizer=Adam(lr=0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])
# Step 5: Data Augmentation and Loading
train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
'path/to/train_data',
target_size=(224, 224),
batch_size=32,
class_mode='categorical'
)
# Step 6: Train the model
model.fit(train_generator, epochs=10) # Adjust the number of epochs as 
needed
# Optionally, you can unfreeze some layers and fine-tune
for layer in base_model.layers[-20:]:
layer.trainable = True
model.compile(optimizer=Adam(lr=0.0001),
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_generator, epochs=5) # Fine-tune for a few more epochs if 
needed

exp 8 Using a pre trained model on Keras for Transfer Learning.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
# Step 1: Load the pre-trained VGG16 model
base_model=tf.keras.applications.VGG16(weights='imagenet', 
include_top=False, input_shape=(224, 224, 3))
# Step 2: Freeze the pre-trained layers
for layer in base_model.layers:
layer.trainable = False
# Step 3: Build a custom classifier on top of the pre-trained model
model = models.Sequential([
base_model,
layers.Flatten(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(1, activation='sigmoid')
# Binary classification, change to num_classes for multi-class
])
# Step 4: Compile the model
model.compile(optimizer=Adam(lr=0.001),
loss='binary_crossentropy', # Change to 'categorical_crossentropy' for multiclass
metrics=['accuracy'])
# Step 5: Data Augmentation and Loading
train_datagen = ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
'path/to/train_data',
target_size=(224, 224),
batch_size=32,
class_mode='binary' # Change to 'categorical' for multi-class
)
test_generator = test_datagen.flow_from_directory(
'path/to/test_data',
target_size=(224, 224),
batch_size=32,
class_mode='binary' # Change to 'categorical' for multi-class
)
# Step 6: Train the model
model.fit(train_generator,
epochs=10,
validation_data=test_generator)
# Optionally, you can unfreeze some layers and fine-tune
for layer in base_model.layers[-4:]:
layer.trainable = True
model.compile(optimizer=Adam(lr=0.0001),
loss='binary_crossentropy', # Change to 'categorical_crossentropy' for multiclass
metrics=['accuracy'])
model.fit(train_generator,
epochs=5,
validation_data=test_generator)


Ex:9 Perform Sentiment Analysis using RNN.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sample data (replace this with your own dataset)
texts = ["This is a positive review.", "Negative sentiment in this one."]
labels = [1, 0] # 1 for positive, 0 for negative
# Tokenization and Padding
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, 
truncating='post', padding='post')
# Build the RNN Model
model = Sequential([
Embedding(input_dim=len(word_index)+1,output_dim=16, 
input_length=max_length),
LSTM(100),
Dense(1, activation='sigmoid')
])
# Compile the Model
model.compile(optimizer='adam',loss='binary_crossentropy', 
metrics=['accuracy'])
# Train the Model
model.fit(padded_sequences, labels, epochs=10)
# Make Predictions
new_texts = ["Another positive example.", "Not happy with this."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences=pad_sequences(new_sequences, 
maxlen=max_length, truncating='post', padding='post')
predictions = model.predict(new_padded_sequences)
# Display the predictions
for text, prediction in zip(new_texts, predictions):
sentiment = "Positive" if prediction > 0.5 else "Negative"
print(f'Text: "{text}"\nPredicted Sentiment: {sentiment}\n')


ex 10 Image generation using GAN

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
# Load and preprocess data (if using a specific dataset)
# ...
# Generator Model
def build_generator(latent_dim):
model = models.Sequential()
model.add(layers.Dense(256, input_dim=latent_dim, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(28*28, activation='sigmoid'))
model.add(layers.Reshape((28, 28, 1)))
return model
# Discriminator Model
def build_discriminator(img_shape):
model = models.Sequential()
model.add(layers.Flatten(input_shape=img_shape))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
return model
# Combined GAN Model
def build_gan(generator, discriminator):
discriminator.trainable = False
model = models.Sequential()
model.add(generator)
model.add(discriminator)
return model
# GAN Parameters
latent_dim = 100
img_shape = (28, 28, 1)
# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer='adam',loss='binary_crossentropy', 
metrics=['accuracy'])
# Build the generator
generator = build_generator(latent_dim)
# Build and compile the GAN model
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')
# Training the GAN
batch_size = 64
epochs = 30000
# Sample and generate images
def generate_fake_samples(generator, latent_dim, n_samples):
noise = np.random.normal(0, 1, (n_samples, latent_dim))
generated_images = generator.predict(noise)
return generated_images
# Training loop
for epoch in range(epochs):
# Train discriminator
real_images = ... # Load real images from the dataset
real_labels = np.ones((batch_size, 1))
fake_images = generate_fake_samples(generator, latent_dim, batch_size)
fake_labels = np.zeros((batch_size, 1))
d_loss_real = discriminator.train_on_batch(real_images, real_labels)
d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
# Train generator
noise = np.random.normal(0, 1, (batch_size, latent_dim))
valid_labels = np.ones((batch_size, 1))
g_loss = gan.train_on_batch(noise, valid_labels)
# Print progress and save generated images
if epoch % 1000 == 0:
print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
# Save generated images
generated_images = generate_fake_samples(generator, latent_dim, 16)
for i in range(16):
plt.subplot(4, 4, i+1)
plt.imshow(generated_images[i, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()
