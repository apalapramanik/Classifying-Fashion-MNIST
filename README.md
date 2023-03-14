# Classifying-Fashion-MNIST

### **Dataset** : Fashion MNIST

Fashion MNIST is a popular image classification dataset used in the field of machine learning and computer vision. It contains 70,000 grayscale images of 28x28 pixels each, representing 10 different types of clothing items such as t-shirts, trousers, and dresses. The dataset is often used as a benchmark for testing and comparing different machine learning algorithms for image classification tasks. It is widely used by researchers, students, and practitioners to explore new techniques and models for image recognition. The Fashion MNIST dataset is considered to be an upgrade to the classic MNIST dataset, which consists of handwritten digits. The dataset is freely available for download and use from the official website. You can find more information about the dataset and download it from the following link: https://github.com/zalandoresearch/fashion-mnist

### **MODELS** :

We design and implement two architectures for this problem. For each architecture, we vary the number and sizes of the layers, but use at least one hidden layer and only fully connected layers with ReLU or a variation (not convolutional nodes) for all hidden nodes and softmax for the output layer. We measure loss with cross-entropy after mapping the class labels to one-hot vectors.

#### Tyler:

This model consists of a simple fully connected neural network with 5 hidden layers, each containing 128 ReLU activation nodes followed by a dropout layer of 0.3. The input layer consists of a Flatten layer that flattens the $28\times28$ image into a 784-dimensional input vector. The output layer contains a fully connected layer with num\_classes nodes and a softmax activation function, which produces a probability distribution over the num\_classes possible classes .The model has been illustrated in Figure below.

![viz1](https://user-images.githubusercontent.com/50993551/224874039-c75419b9-e775-48f8-8fab-14ede8831545.png)




#### Jackson:

 The input to the model is a $28\times28$ grayscale image (represented as a 2D array), which is flattened into a 1D array of 784 elements by the Flatten layer.

The first hidden layer has 512 nodes, followed by a dropout layer with a dropout rate of 0.3. The next three hidden layers have 256, 128, and 64 nodes, respectively, each followed by a dropout layer with a dropout rate of 0.3. These layers progressively decrease the number of nodes in the network, which can help prevent overfitting by reducing the model's capacity.The output layer has num\_classes nodes, where num\_classes is the number of classes in the Fashion-MNIST dataset (10 in this case). The softmax activation function is used to compute the class probabilities for each input image.The kernel\_regularizer argument in the Dense layers specifies L2 regularization with a strength of 0.001, if use\_regularizer is set to True. L2 regularization penalizes large weights in the network, which can help prevent overfitting.Overall, this model has a total of 4 hidden layers, each followed by a dropout layer, and an output layer. The use of dropout and L2 regularization should help prevent overfitting and improve generalization performance.The model has been illustrated in Figure below.

![viz2](https://user-images.githubusercontent.com/50993551/224874297-c8b45b9d-cf41-4b21-bea1-8135685eb1fa.png)


### Hyperparamter Tuning for Tyler:

To evaluate the performance of the first model Tyler,  with and without regularization, we train the model with various hyperparameter settings and plot the accuracy on a validation set. We  choose four hyperparameters to vary, such as the dropout rate, the learning rate, the batch size, and the number of epochs. For each hyperparameter setting, we train the model with and without L2 regularization and record the validation accuracy as shown in the plots below.

<img width="459" alt="plots" src="https://user-images.githubusercontent.com/50993551/224874937-729d394e-4d79-4f23-867b-9b8dfeeea015.png">

### Model Performance on Test Dataset:

We chose a final dropout rate of 0.3, batch size of 128, learning rate of 0.001 and train each model for 50 epochs to get a final accuracy of approximatesly ~88%. The accurate results are listed in the table given below.

<img width="388" alt="table" src="https://user-images.githubusercontent.com/50993551/224875142-4d9564ed-bebc-4b46-92e0-a5e92ba8a60c.png">

### Confusion Matrix for each model:

A confusion matrix is a valuable tool for interpreting the results of a classification model. It provides a clear and concise representation of the model's performance, allowing us to identify areas where it excels and areas where it struggles. By comparing the actual and predicted values, the confusion matrix allows us to calculate important performance metrics such as accuracy, precision, recall, and F1 score. These metrics help us to understand how well the model is able to correctly identify the different classes in the dataset. By visualizing the performance of the model in the form of a confusion matrix in the figure below,, we can gain insights into the strengths and weaknesses of the model and make informed decisions about how to improve its performance.

<img width="500" alt="cm" src="https://user-images.githubusercontent.com/50993551/224875343-13b617a7-6cd0-4de6-a076-0372d29beb81.png">


Program Files:

Three major program files for this project, with the following names:
1. main-FMNIST.py : Code that runs the main loop of training the TensorFlow models
2. model-FMNIST.py : TensorFlow code that defines the network
3. util-FMNIST.py : Helper functions (e.g., for loading the data, small repetitive functions)

Models saved:

1. model_tyler_0.h5 : model tyler without regularizer
2. model_tyler_1.h5 : model tyler with regularizer
3. model_jackson_0.h5 : model jackson without regularizer
4. model_jackson_1.h5 : model jackson with regularizer



