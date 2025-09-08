## 🌰🥐🥨 Support Vector Regression 🥨🥐🌰
As mentioned at the beginning of the module, apart from being able to solve classification problems, support vectors can also be used to predict continuous data, namely regression cases. Support Vector Regression (SVR) uses the same principles as SVM in the classification case. The difference is that if in the case of classification, SVM tries to find the largest 'path' that can separate samples from different classes, then in the case of regression SVR tries to find a path that can accommodate as many samples as possible in the 'path'. Look at the following image to see an example of SVR.

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/42f3348d-4e03-40ab-af42-8be5211e57f7)


As explained by PCMag in Hands-On Machine Learning with Scikit Learn [4], the image above shows two linear SVM Regression models trained on some random linear data, one with a large margin (ϵ = 1.5) and the other with a small (ϵ = 0.5). The road width is controlled by the hyperparameter ϵ, which is also called the maximum error. Adding training data into the margin will not affect the model predictions. Therefore, the model is referred to as ϵ-insensitivity. In contrast to SVM where the support vector is 2 samples from 2 different classes that have the closest distance, in SVR the support vector is a sample that is a road divider that can accommodate all samples in the data. M. Awad and R. Khanna in chapter 4 of their book illustrate the support vector in SVR as follows.

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/1f2fd65f-f38a-4110-a015-1af0057398bc)

Let's take an example of implementing SVR in the case of house price prediction in the city of Boston, United States, using a very popular dataset for regression cases: "Boston Housing Price.csv". First, we will see how a simple linear regression predicts this data, then we will compare the results with SVR. Recalling a little about linear regression which was discussed in the previous module, the performance measure for linear regression problems is the Root Mean Square Error (RMSE). RMSE gives an idea of how much error there is in the predictions made by the system. The goal, of course, is to get the minimum possible error or error rate. In the case of house price predictions in Boston, linear regression will give the following plot results.

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/56b751f4-852e-4100-9c2c-2e78a3d872ac)


Let's try to implement SVR on the same dataset. One of the advantages of SVR over linear regression is that it gives us the flexibility to determine how much error is acceptable in our model. The SVR algorithm will find a suitable line (hyperplane) to fit the data. We can adjust the parameter ϵ to get the model accuracy we want. If we choose the value ϵ = 5 and we plot the data, the results are as follows.

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/dc2eb4d7-f5d0-4d24-925a-eddfca5618b8)

The red line in the figure shows the regression line, while the blue line shows the margin of error, ϵ, which we set earlier to ϵ = 5 (or on a scale of thousands that means $5,000). From the image above you can probably immediately see that the SVR algorithm cannot provide good prediction results for all data because some points are still outside the limits. Therefore, we need to add another parameter to the algorithm, namely parameter C which is called the regularization parameter or regularity parameter. There are also those who call it slack parameters. Jui Yang Hsia and Chih-Jen Lin in their writing state that this regularization parameter functions to avoid overfitting in the training data. Returning to the case of house predictions in Boston, let's try adding parameter C to the data. If we set the value C = 2.0, then the results are as follows.

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/d5e57c58-92ed-4492-a307-09d64fed6bd1)

Note that now our model fits the distribution of the data better than the previous model.

## 🌰🥐🥨 Reading Dataset 🥨🥐🌰
The dataset we will use is data about a person's length of service and salary.
```
import pandas as pd
```
 
## 🌰🥐🥨 Turning It Into a Dataframe 🥨🥐🌰
```
data = pd.read_csv('Salary_Data.csv')
```

## 🌰🥐🥨 Looking for Missing Value 🥨🥐🌰
Next, we can see whether there are missing values in the dataset with the .info() function. The output from the cell below shows that there are no missing values in the dataset.
```
data.info()
```

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/b458cb0c-64a6-455f-984a-08b0aad8e4bf)

## 🌰🥐🥨 Cleaning Dataframes 🥨🥐🌰
Next we display the first 5 rows of the dataframe.
```
data.head()
```

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/ccbcdc2c-d480-449d-a948-3dab89daec98)

## 🌰🥐🥨 Separate Between Attributes and Labels 🥨🥐🌰
Then we separate the attributes and labels that we want to predict. When there is only one attribute in the dataframe, the attribute needs to be changed so that it can be accepted by the model from the SKLearn library. To change the shape of the attribute we need the numpy library.
```
import numpy as np
X = data['YearsExperience']
Y = data['Salary']
X = X[:,np.newaxis]
```

## 🌰🥐🥨 Object Support Vector Regression 🥨🥐🌰
Next we create a support vector regression object and here we will try to use the parameters C = 1000, gamma = 0.05, and the kernel 'rbf'. After the model is created we will train the model with a fit function on the data.
```
from sklearn.svm import SVR
```
 
## 🌰🥐🥨 Model Parameters 🥨🥐🌰
Build a model with C, gamma, and kernel parameters
```
model = SVR(C=1000, gamma=0.05, kernel='rbf')
```
 
## 🌰🥐🥨 Fit Function 🥨🥐🌰
Training a model with a fit function
```
model.fit(X,y)
```

## 🌰🥐🥨 Model Visualization 🥨🥐🌰
Finally, we can visualize how our SVR model adapts to patterns in the data using the matplotlib library.
```
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model.predict(X))
```

## 🌰🥐🥨 Output Visualization 🥨🥐🌰
The visualization results show that the model we developed was not able to adapt to the patterns in the data well.

![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/a010fd10-a93b-448a-be56-79fe92574c43)

## 🌰🥐🥨 License 🥨🥐🌰
- Copyright by Diantya Pitaloka
