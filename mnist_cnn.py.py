import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 # stands for Computer Vision, it is the OpenCV
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

type(X_train)

# Reshape the input data to have a channel dimension (grayscale)
X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

#shape of the numpy arrays
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

#printing the 10th image from training set
print(X_train[10])

plt.imshow(X_train[500])
plt.show()

print(Y_train[500])

#unique values in Y_train
print(np.unique(Y_train))

#unique values in Y_test
print(np.unique(Y_test))

# Normalize the pixel values to the range of [0, 1]
X_train = X_train / 255
X_test = X_test / 255

# Building and setting up the layers of our CNN
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling the CNN
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the CNN to the training data
model.fit(X_train, Y_train, epochs=10)

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

plt.imshow(X_test[0])
plt.show()

print(Y_test[0])

y_pred = model.predict(X_test) #model.predict() gives the prediction probability for each class for that data point

print(y_pred.shape)

print(y_pred[0])

#converting the prediction probabilities to class labels
label_for_first_test_image = np.argmax(y_pred[0])
print(label_for_first_test_image)

#converting the prediction probabilities to class labels for all test data points
#these will be the final predictions made by our model
y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels)

cm = confusion_matrix(Y_test, y_pred_labels)
print(cm)

plt.figure(figsize=(15, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.ylabel('True labels')
plt.xlabel('Predicted labels')

#Building a predictive system
input_image_path = '/content/MNIST_digit.png'

input_image = cv2.imread(input_image_path)

type(input_image)

cv2_imshow(input_image)

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

grayscale.shape

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize.shape

# Normalize the pixel values to the range of [0, 1]
image_reshaped = image_reshaped / 255

# Reshape the input data to have a channel dimension (grayscale)
image_reshaped = np.reshape(input_image_resize, (1, 28, 28, 1))

# Make the prediction
input_prediction = model.predict(image_reshaped)
print(image_reshaped)

# Get the predicted class label
predicted_label = np.argmax(input_prediction)
print(predicted_label)