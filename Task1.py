import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.metrics import r2_score

#load data into model from url
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
data.head()

#check data headers
print(data.head())

#Organise data and define training set
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Linear regression training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Test prediction
y_pred = regressor.predict(X_test)

#visualize results
plot.scatter(X_test, y_test, color='red')
plot.plot(X_test, y_pred, color='blue')
plot.title('Hours vs Scores (Test set)')
plot.xlabel('Hours')
plot.ylabel('Scores')
plot.show()

#get r2 score
accuracy = r2_score(y_test, y_pred)
print(accuracy)

# hours input for testing
hours = float(input("Enter the number of hours: "))

# convert hours to a numpy array and reshape it
hours_array = np.array(hours).reshape(-1, 1)

# use the trained model to make a prediction
predicted_score = regressor.predict(hours_array)

print("Predicted score: {:.2f}".format(predicted_score[0]))