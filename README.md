<h1 align="center">MNIST Handwritten Digit Classification Using CNN</h1>



![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Description
The MNIST dataset, which stands for Modified National Institute of Standards and Technology, consists of 60,000 grayscale images of handwritten digits ranging from 0 to 9. It can be easily imported from  `Keras Datasets` using the command    `from keras.datasets import mnist` for the purpose of classifying these images into their corresponding digit classes.

## Dataset
The dataset can be found and is described <a href="https://keras.io/api/datasets/mnist/">here</a>.

## Model Architecture
The CNN model is designed to capture spatial patterns in the input images using convolutional layers. The architecture of the model is as follows:

Input layer: Accepts input images of size 28x28 pixels.

Convolutional layers: Apply a set of filters to extract relevant features from the input images. The number of filters and their dimensions can be adjusted based on the complexity of the problem.

Pooling layers: Downsample the feature maps obtained from the convolutional layers, reducing the spatial dimensions while preserving important information.

Fully connected layers: Flatten the feature maps and feed them into a fully connected neural network for classification.

Output layer: Produces a probability distribution over the 10 possible classes (digits 0-9) using a softmax activation function.

## Output
![Screenshot (215)](https://github.com/PranavBawgikar/MNIST-digit-recognizer/assets/102728016/91236571-c04e-4382-a89c-b4cc81921280)
![Screenshot (214)](https://github.com/PranavBawgikar/MNIST-digit-recognizer/assets/102728016/1e8b2377-4991-4c18-a5dd-d8cbe971e095)<br><br>
![Screenshot (213)](https://github.com/PranavBawgikar/MNIST-digit-recognizer/assets/102728016/d7a1d587-8df2-4718-b314-6f3e6a466767)


## Results
![Screenshot (212)](https://github.com/PranavBawgikar/MNIST-digit-recognizer/assets/102728016/880a3d88-0b71-45e7-85d3-3b43dbc57244)


Feel free to modify this template to fit your specific project, and let me know if you have any questions or need further assistance!
