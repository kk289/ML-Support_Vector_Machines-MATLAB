# Machine Learning (MATLAB) - Support Vector Machines

Machine Learning course from Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/home/week/7).

### Environment
- macOS Catalina (version 10.15.3)
- MATLAB 2018 b

### Dataset
- ex6data1.mat
- ex6data2.mat
- ex6data3.mat
- spamTrain.mat
- spamTest.mat
- emailSample1.txt
- emailSample2.txt
- spamSample1.txt
- spamSample2.txt
- vocab.txt

### Files included in this repo
- ex6.m - Octave/MATLAB script for the first half of the exercise 
- ex6data1.mat - Example Dataset 1
- ex6data2.mat - Example Dataset 2
- ex6data3.mat - Example Dataset 3
- svmTrain.m - SVM training function
- svmPredict.m - SVM prediction function 
- plotData.m - Plot 2D data 
- visualizeBoundaryLinear.m - Plot linear boundary 
- visualizeBoundary.m - Plot non-linear - boundary 
- linearKernel.m - Linear kernel for SVM

[⋆] gaussianKernel.m - Gaussian kernel for SVM

[⋆] dataset3Params.m - Parameters to use for Dataset 3

- ex6_spam.m - Octave/MATLAB script for the second half of the exercise
- spamTrain.mat - Spam training set
- spamTest.mat - Spam test set
- emailSample1.txt - Sample email 1
- emailSample2.txt - Sample email 2
- spamSample1.txt - Sample spam 1
- spamSample2.txt - Sample spam 2
- vocab.txt - Vocabulary list
- getVocabList.m - Load vocabulary list
- porterStemmer.m - Stemming function
- readFile.m - Reads a file into a character string
- submit.m - Submission script that sends your solutions to our servers 

[⋆] processEmail.m - Email preprocessing

[⋆] emailFeatures.m - Feature extraction from emails

## Part 1: Support Vector Machines
We will be using support vector machines (SVMs) with various example 2D datasets. Experimenting with these datasets will help us gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs.

### Part 1.1: Example Dataset 1
In this dataset, the positions of the positive examples (indicated with +) and the negative examples (indicated with o) suggest a natural separation indicated by the gap.
```
ex6data1.mat
```

![dataset1](Figure/dataset1.jpg)
- Figure: Example Dataset 1: SVM decision boundary

We will try using different values of the C parameter with SVMs. Informally, the C parameter is a positive value that controls the penalty for misclassified training examples. A large C parameter tells the SVM to try to classify all the examples correctly. C plays a role similar to 1/λ, where λ is the regularization parameter that we were using previously for logistic regression.

![dataset1](Figure/example1db.jpg)
- Figure: SVM decision boundary with C = 1 (Dataset 1)

When C = 1, we find that the SVM puts the decision boundary in the gap between the two datasets and misclassifies the data point on the far left.

![dataset1](Figure/example2db.jpg)
- Figure: SVM decision boundary with C = 100 (Dataset 1)

When C = 100, we find that the SVM now classifies every single example correctly, but has a decision boundary that does not appear to be a natural fit for the data.

### Part 1.2: SVM with Gaussian Kernels
We will be using SVMs to do non-linear classification. In particular, we will be using SVMs with Gaussian kernels on datasets that are not linearly separable.

#### Part 1.2.1: Gaussian kernel
To find non-linear decision boundaries with the SVM, we need to first implement a Gaussian kernel.

The Gaussian kernel as a similarity function that measures the “distance” between a pair of examples, (x(i),x(j)). The Gaussian kernel is also parameterized by a bandwidth parameter, σ, which determines how fast the similarity metric decreases (to 0) as the examples are further apart.

The Gaussian Kernal function: 
![gaussian](Figure/gaussian.png)

##### gaussianKernel.m
```
% Gaussian kernel for SVM
diff = sum((x1 - x2).^2);
temp = 2 * (sigma.^2);
sim = exp(-(diff/temp));
```

Result: 
Gaussian Kernel between (x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2.000000)-> *0.324652*




## Course Links 

1) Machine Learning by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/home/week/7).

2) [Regularized Linear Regression](https://www.coursera.org/learn/machine-learning/home/week/6)
(Please notice that you need to log in to see the programming assignment.) #ML-Support_Vector_Machines-MATLAB