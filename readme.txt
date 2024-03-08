# Sudoku solving using image processing in python	



# Motivation : 

This code is a Python script that demonstrates how to build and train a convolutional neural network (CNN) on the MNIST dataset using Keras. The code loads the dataset, preprocesses the input data, builds a CNN model, compiles the model with an optimizer, loss function, and metrics, trains the model on the training set, evaluates the model on the test set, and plots the accuracy and loss curves for the training and validation sets.

The code also includes an example of how to load and display an image using OpenCV and Matplotlib. The example loads an image from a file, saves it to a different file, reads the saved file, and displays the image using Matplotlib.




# Tech/Framework used

This is a convolutional neural network (CNN) model trained on the MNIST dataset for digit recognition. The model takes in grayscale images of size 28x28 pixels as input and has a sequence of layers that extract features from the input images and classify them into one of 10 possible classes (digits 0 to 9).

The model architecture consists of two convolutional layers followed by max pooling layers, a flattening layer, and two fully connected (dense) layers. The first convolutional layer has 32 filters and a filter size of 3x3, while the second convolutional layer has 64 filters and a filter size of 3x3. The first dense layer has 64 units and uses the relu activation function, and the final dense layer has 10 units (one for each digit) and uses the softmax activation function.

The model is trained using the rmsprop optimizer, the categorical_crossentropy loss function, and the accuracy metric. It is trained for 7 epochs with a batch size of 64.

The accuracy and loss curves for the training and validation sets are plotted using Matplotlib. Additionally, a random image from the training set is plotted using Matplotlib as well.

Lastly, OpenCV is used to read in an image of a Sudoku puzzle, which is then displayed using Matplotlib. 





# Features 

1. Model type: Convolutional Neural Network (CNN)


2. Dataset: MNIST (handwritten digit recognition)


3. Input data: grayscale images of size 28x28 pixels


4. Architecture:
      •	Two convolutional layers with 32 and 64 filters respectively, and 3x3 filter size
      •	Max pooling layers after each convolutional layer
      •	Flattening layer
      •	Two fully connected (dense) layers with 64 and 10 units respectively
      •	Relu activation function for the first dense layer
      •	Softmax activation function for the final dense layer


5. Training:
      •	Optimizer: rmsprop
      •	Loss function: categorical_crossentropy
      •	Accuracy metric
      •	Trained for 7 epochs with a batch size of 64


6. Evaluation:
      •	Accuracy and loss curves plotted for training and validation sets using Matplotlib


7. Visualization:
      •	A random image from the training set is plotted using Matplotlib
      •	An image of a Sudoku puzzle is read in using OpenCV and displayed using Matplotlib.




# Installation

The code is an implementation of a convolutional neural network (CNN) model to classify handwritten digits from the MNIST dataset. The code starts by loading the dataset and preprocessing it by normalizing the pixel values and reshaping the images to include a channel dimension for the convolutional layers. Then, the labels are converted to one-hot encoding.

Next, the CNN model is built using the Sequential model object from Keras. The model includes a 2D convolutional layer, followed by a max pooling layer, another convolutional layer, and a flattening layer. Finally, two fully connected layers are added, and the output layer uses the softmax activation function to produce a probability distribution over the possible digit classes.

The model is then compiled with the rmsprop optimizer, categorical_crossentropy loss function, and accuracy metric, and trained for 7 epochs using a batch size of 64. After training, the model is evaluated on the test set, and the test accuracy is printed.

Finally, the code plots the accuracy and loss curves for the training and validation sets and displays a random image from the training set.

Additionally, the code loads an image of a Sudoku puzzle using OpenCV, saves it as a new file, and displays the image using matplotlib. 






# Usage

The code is a script that trains a convolutional neural network (CNN) on the MNIST dataset of handwritten digits. It uses the Keras library to define and train the model. The MNIST dataset consists of 28x28 grayscale images of handwritten digits, along with labels indicating the true digit value (0-9).

The code first loads the dataset and preprocesses the input data by scaling the pixel values to a range between 0 and 1. It also reshapes the images to include a channel dimension (for the convolutional layers) and converts the labels to one-hot encoding.

The model architecture consists of two convolutional layers followed by a flattening layer and two fully connected (Dense) layers. The first convolutional layer has 32 filters and a filter size of 3x3, and the second has 64 filters and a filter size of 3x3. Both convolutional layers use the ReLU activation function. The model also includes max pooling layers after each convolutional layer, which reduce the spatial size of the feature maps by a factor of 2. The final fully connected layer has 10 units (one for each digit) and uses the softmax activation function to produce a probability distribution over the possible digit classes.

The model is trained using the categorical_crossentropy loss function and the rmsprop optimizer. The performance of the model is evaluated on the test set, and the test accuracy is printed. The script also plots the accuracy and loss curves for the training and validation sets.

Finally, the script randomly selects an image from the training set, plots it, and saves an image of a Sudoku puzzle to be used for digit recognition. 

This is a Python script that trains a convolutional neural network (CNN) on the MNIST handwritten digit recognition dataset using the Keras library. The code first loads and preprocesses the dataset, then defines the architecture of the CNN model and compiles it. The model is trained on the training set and evaluated on the test set, and the test accuracy is printed to the console. Finally, the script plots the training and validation accuracy and loss curves.

To run this code, you will need to have Keras and its dependencies installed, as well as the MNIST dataset. You can install Keras using pip or conda, and you can download the MNIST dataset using Keras' built-in functions (as shown in the code). Once you have these installed and downloaded, you can simply run the script in a Python environment (such as Anaconda or Jupyter Notebook).

This code is an implementation of a Convolutional Neural Network (CNN) model for image classification using the MNIST dataset. The MNIST dataset is a set of handwritten digit images, consisting of 60,000 training images and 10,000 test images.

The code first loads the MNIST dataset and preprocesses the input data by normalizing the pixel values to a range between 0 and 1, reshaping the images to include a channel dimension (for the convolutional layers), and converting the labels to one-hot encoding.

The CNN model is built using the Keras library, which allows us to build a neural network model layer-by-layer in a sequential manner. The model consists of several layers including a 2D convolutional layer, a max pooling layer, a flattening layer, and several fully connected layers. The final layer uses the softmax activation function to produce a probability distribution over the possible digit classes.

The model is then compiled using the categorical_crossentropy loss function, the rmsprop optimizer, and the accuracy metric. The model is trained on the training set for 7 epochs with a batch size of 64, and evaluated on the test set. Finally, the code plots the accuracy and loss curves for the training and validation sets.






# Discription of this code :


Python script that performs the following steps:

1. Loads an image of a Sudoku puzzle from a file.

2. Preprocesses the image to reduce noise and threshold the image adaptively.

3. Identifies the biggest contour in the image, which should correspond to the outline of the puzzle.

4. Aligns the image so that the puzzle is a perfect square.

5. Divides the image into 81 smaller cells, each containing one of the digits of the puzzle.

6. Resizes each cell to 28x28 pixels and passes it through a pre-trained neural network to predict the digit it contains.

7. Prints the predicted digits for each cell.



Based on the code, it appears that the pre-trained neural network is not included, so it cannot be determined how accurate the digit predictions will be.

This code is a Python implementation of a Sudoku solver using a backtracking algorithm.

The code begins by defining the input Sudoku grid as a 3D NumPy array. It then finds the indices of any unequal elements in this array, which is used to check for prediction errors later.

Next, the code defines the possible function which checks whether a given number can be placed in a given cell of the Sudoku grid. It checks if the number appears in the same row, column, or 3x3 square as the cell being checked.

The solve function is then defined. It loops through each cell in the grid and checks if it is empty (represented by the number 0). If it is empty, it tries to place a number from 1 to 9 in the cell using the possible function. If a number can be placed in the cell, the function calls itself recursively to continue solving the Sudoku grid. If no number can be placed in the cell, it backtracks to the previous cell and tries a different number.

Once the Sudoku has been solved, the solve function converts the grid into an image using the OpenCV library. It draws the Sudoku grid and the solved digits in the corresponding boxes.

Finally, the solve function is called to solve the input Sudoku grid and generate the output image. The output image is saved as sudoku.jpg.



# Team Member :

      •	21BCE117 - Keertan Parikh
      •	21BCE119 - Meet Khunt
      •	21BCE132 - Darshil Ladani

