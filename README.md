# Image_classifiaction

Using classic Neural Networks(CNN) to perform image classification. Using Keras and TensorFlow in Python. The labels were pre-defined as the class names and the model was trained on this neural network.

## Dataset Preparation: 
Collect or obtain a dataset of labeled images for training and testing your image classification model. Ensure that the dataset is appropriately labeled with the corresponding class or category for each image.
Accessing dataset from the link - http://www.vision.caltech.edu/Image_Datasets/Caltech101

## Data Preprocessing: 
Preprocess the image data to ensure it is in a suitable format for training. Common preprocessing steps include resizing images to a consistent size, normalizing pixel values, and splitting the dataset into training and testing sets.

## Model Architecture Design: 
Define the architecture of your image classification model using the Keras API, which is built on top of TensorFlow. Choose a suitable model architecture such as Convolutional Neural Networks (CNNs) that are well-suited for image classification tasks.

## Model Compilation: 
Compile the model by specifying the optimizer, loss function, and evaluation metrics. The optimizer determines the learning algorithm used to update the model's weights during training, while the loss function measures the model's performance. Common optimizers include Adam, SGD, and RMSprop, while categorical cross-entropy is commonly used as the loss function for multi-class classification.

### Insight
### Comparison of accuracy of different activation functions

![image](https://user-images.githubusercontent.com/35174083/55663673-68e2b900-57ef-11e9-8e8c-b43badef6c41.png)
 

![image](https://user-images.githubusercontent.com/35174083/55663685-9e87a200-57ef-11e9-9e05-94450591cf5e.png)


### Conclusion
1. T-1 performs better for image classification as compared to T-2 activation function
2. The convolutional network gives an accuracy of 95% for the 10 classes with maximum number of images 
3. While training the images using CNN the number of training samples in important. For example, if there are less samples to train on then the model won't perform accurately.
