## Support Vector Regression

## Reading Dataset
The dataset we will use is data about a person's length of service and salary.
- import pandas as pd
 
## Turning It Into a Dataframe
- data = pd.read_csv('Salary_Data.csv')

## Looking for Missing Value
Next, we can see whether there are missing values in the dataset with the .info() function. The output from the cell below shows that there are no missing values in the dataset.
- data.info()
![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/b458cb0c-64a6-455f-984a-08b0aad8e4bf)

## Cleaning Dataframes
Next we display the first 5 rows of the dataframe.
- data.head()
![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/ccbcdc2c-d480-449d-a948-3dab89daec98)

## Separate Between Attributes and Labels
Then we separate the attributes and labels that we want to predict. When there is only one attribute in the dataframe, the attribute needs to be changed so that it can be accepted by the model from the SKLearn library. To change the shape of the attribute we need the numpy library.
- import numpy as np
- X = data['YearsExperience']
- Y = data['Salary']
- X = X[:,np.newaxis]

## Object Support Vector Regression
Next we create a support vector regression object and here we will try to use the parameters C = 1000, gamma = 0.05, and the kernel 'rbf'. After the model is created we will train the model with a fit function on the data.
- from sklearn.svm import SVR
 
## Model Parameters
Build a model with C, gamma, and kernel parameters
model = SVR(C=1000, gamma=0.05, kernel='rbf')
 
## Fit Function
Training a model with a fit function
model.fit(X,y)

## Model Visualization
Finally, we can visualize how our SVR model adapts to patterns in the data using the matplotlib library.
- import matplotlib.pyplot as plt
- plt.scatter(X, y)
- plt.plot(X, model.predict(X))

## Output Visualization
The visualization results show that the model we developed was not able to adapt to the patterns in the data well.
![image](https://github.com/diantyapitaloka/Support-Vector-Regression/assets/147487436/a010fd10-a93b-448a-be56-79fe92574c43)
