# Distracted-Driver-Detection-UsingML
Driving a car is a complex task, and it requires complete attention. Distracted driving is any activity that 
takes away the driver’s attention from the road. Approximately 1.35 million people die each year because 
of road traffic crashes.
Road traffic crashes cost most countries 3% of their gross domestic product So, our aim/goal in this 
project is to detect if the car driver is driving safe or performing any activity that might result in an 
accident or any harm to others, by using various Machine Learning Models to classify the provided 
images into different categories of Distraction.

# Dataset - 
The dataset used is State Farm Distracted Driver Detection taken from https://www.kaggle.com/c/state-farm-distracted-driver-detection/data.
The dataset has 22424 training images, 79727 testing images and has 10 classes. The images are coloured and are of size 640×480 pixels. The classes are labelled as follows:
c0: safe driving, c1: texting — right, c2: talking on the phone — right, c3: texting — left, c4: talking on the phone — left, c5: operating the radio, c6: drinking, 
c7: reaching behind, c8: hair and makeup, c9: talking to a passenger
DATASET VISUALIZATION-
![image](https://user-images.githubusercontent.com/81475333/123747805-ffb85600-d8d0-11eb-9ca8-4af07fd8568e.png)

# Data Pre-processing
Images are resized to 64 × 64 × 3 using CV2 in order to improve the computing efficiency of the classifier.
Stratified splitting is used to split the dataset into 80:20 Training-Testing ratio. The training dataset is further split into 90:10 Training-Validation set.
Thus, the final training set has 16145 images; the final validation set has 1794 images and the final testing set has 4485 images.

# Workflow

![image](https://user-images.githubusercontent.com/81475333/123747962-3b532000-d8d1-11eb-979a-cba25926b9a1.png)

# Feature Extraction

Following feature extraction techniques were used for extracting features from the images:
1.HOG — The histogram of oriented gradients is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image.
2.LBP — Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the pixels of an image by thresholding the neighbourhood of each pixel and considers the result as a binary number
3.Color Histogram-
A color histogram is a representation of the distribution of colors in an image. For digital images, a color histogram represents the number of pixels that have colors in each of a fixed list of color ranges, that span the image’s color space.
4.KAZE -
KAZE is a 2D feature detection and description method that operates completely in nonlinear scale space.Features extracted using Feature Extraction Techniques

# Normalization: Min-Max Normalization - 
In this technique of data normalization, a linear transformation is performed on the original data. Minimum and maximum value from data is fetched and each value is replaced according to the following formula:
x' = (x - Am)/σA
where x' = normalized value
      x  = original value
      Am = mean of dataset
      σA = standard deviation
      
# Dmensionality Reduction - 
We have used three dimensionality reduction techniques which are stated below:
1. PCA:
   Principal component analysis (PCA) is a technique for reducing the dimensionality of datasets, increasing interpretability but at the same time minimizing information loss.
2. LDA:
   Linear discriminant analysis (LDA) is a generalization of Fisher’s linear discriminant, a method used in statistics and other fields, to find a linear combination of features that characterizes or separates two or more classes of objects or events.
3. LDA over PCA
   LDA is applied on combined features which are obtained after applying PCA to further reduce the features, get a better class separation and to increase computational efficiency.
   
# Traditional ML Models - 
KNN - 
The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other KNN captures the idea of similarity (sometimes called distance, proximity, or closeness)

# Ensembling Methods - 
1.XGBoost - 
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks.
2.Bagging - 
Bootstrap aggregating also called bagging (from bootstrap aggregating), is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting.
3.ADABoost -
AdaBoost algorithm, short for Adaptive Boosting, is a Boosting technique that is used as an Ensemble Method in Machine Learning. It is called Adaptive Boosting as the weights are re-assigned to each instance, with higher weights to incorrectly classified instances.

# Optimal parameters - 
![image](https://user-images.githubusercontent.com/81475333/123750775-9cc8be00-d8d4-11eb-8118-d031c98ad362.png)

# Accuracy obtained - 
![image](https://user-images.githubusercontent.com/81475333/123750846-ae11ca80-d8d4-11eb-9e33-0b1a43882777.png)
