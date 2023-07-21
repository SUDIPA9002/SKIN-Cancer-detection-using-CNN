# SKIN-Cancer-detection-using-CNN

DATASET USED : https://challenge.isic-archive.com/data/#2020. In this dataset, there are 33,126 images of benign and malignant skin cancer

In this folder we will explore the development of a Skin Cancer Detection Model using Convolutional Neural Networks (CNNs) and its deployment using Flask, a popular Python web framework. Skin cancer is one of the most common types of cancer, with early detection being crucial for successful treatment. In recent years, advances in artificial intelligence have opened up new possibilities in medical diagnostics.


## Steps to be followed:
#### Understanding Convolutional Neural Networks (CNNs)
CNNs are a class of deep learning models specifically designed to process and analyze visual data such as images. They are widely used in computer vision tasks, including image classification. The architecture of a CNN involves convolutional layers, pooling layers, and fully connected layers, which work together to learn meaningful features from the input images.

#### Collecting and Preprocessing the Data
To build our skin cancer detection model, we need a dataset of skin lesion images. The ISIC (International Skin Imaging Collaboration) archive is a reliable source for dermatology images. The data should be split into training, validation, and test sets.

Before feeding the images into the CNN, some preprocessing steps include resizing the images to a uniform size, normalizing pixel values, and applying data augmentation techniques to increase the diversity of the training data.

#### Building the CNN Model
The CNN architecture can be customized based on the problem at hand. It typically consists of multiple convolutional layers, each followed by a ReLU activation function and pooling layers for downsampling. Towards the end, fully connected layers are used for classification. Transfer learning can also be employed by using a pre-trained model (e.g., VGG, ResNet) and fine-tuning it on the skin cancer dataset.

The model is trained using a loss function (such as categorical cross-entropy for multi-class classification) and an optimizer (commonly, Adam) to minimize the loss and update the model's weights.

#### Evaluating the Model
After training, the model's performance is evaluated on the validation set. Common evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrix. These metrics help assess the model's ability to correctly identify malignant and benign lesions.

#### Deployment using Flask
Flask is a lightweight and easy-to-use web framework in Python. We will use Flask to build a simple web application that allows users to upload a skin lesion image and receive the model's prediction on whether the lesion is malignant or benign.

#### The deployment process involves the following steps:
1. Save the trained CNN model along with any required preprocessing steps.
2. Create a Flask web application with a route that handles image uploads.
3. Implement a function to preprocess the uploaded image and pass it through the trained model to get the prediction.
4. Return the prediction to the user.
5. User Interface Design


## Conclusion
In this blog, we explored the development of a Skin Cancer Detection Model using CNNs and deploying it with Flask. Early detection of skin cancer using AI-based models can potentially save lives by providing timely and accurate diagnoses. By leveraging the power of deep learning and web frameworks like Flask, we can build applications that bridge the gap between cutting-edge research and real-world medical applications.
