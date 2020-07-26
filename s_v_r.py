#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
Y=Y.reshape(len(Y),1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_Y=StandardScaler()
Y=sc_Y.fit_transform(Y)

#Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y.ravel())

#Predicting a new result
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y),color='red')
plt.plot(sc_X.inverse_transform(X),sc_Y.inverse_transform(regressor.predict(X)),color='blue')
plt.title('Truth or Bluff(Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising the SVR results(for higher resolution and smoother curve)
X_grid=np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y),color='red')
plt.plot(X_grid ,sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid))),color='blue')
plt.title('Truth or Bluff(Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# =========Alternate Method====================================================================
# #Visualising the SVR results(for higher resolution and smoother curve)
# X_grid=np.arange(min(X),max(X),0.1)
# X_grid=X_grid.reshape((len(X_grid),1))
# plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y),color='red')
# plt.plot(sc_X.inverse_transform(X_grid),sc_Y.inverse_transform(regressor.predict(X_grid)),color='blue')
# plt.title('Truth or Bluff(Support Vector Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()
# =============================================================================



