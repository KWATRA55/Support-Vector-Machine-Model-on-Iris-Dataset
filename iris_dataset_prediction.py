import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score


iris= datasets.load_iris()

#split it into features and labels

x= iris.data
y= iris.target

#split the training and testing data

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

#make the model and fit the training data into it
model=svm.SVC()

model.fit(x_train, y_train)

predictions= model.predict(x_test)
acc= accuracy_score(y_test, predictions)

print(predictions)
print(acc)
print(y_test)




