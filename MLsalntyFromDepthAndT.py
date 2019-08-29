import numpy as np
import matplotlib.pyplot as plt
import math
from pandas import DataFrame, read_csv
from mpl_toolkits import mplot3d

bottle = read_csv('bottle.csv',usecols=['Depthm','Salnty','T_degC'])
bottle=bottle.head(1500)

parameters = ['Depthm','T_degC']
objective = 'Salnty'
x0_real = bottle[parameters[0]]
x1_real = bottle[parameters[1]]
y_real = bottle[objective]
X0 = np.array(x0_real)
X1 = np.array(x1_real)
Y1 = np.array(y_real)
i=0
while(i<len(Y1)):
	if math.isnan(Y1[i])|math.isnan(X1[i]):
		Y1=np.delete(Y1,i)
		X1=np.delete(X1,i)
		X0=np.delete(X0,i)
		i=i-1
	i=i+1
X = np.array([X0 ,X1]).T
y = np.array([Y1]).T


class linear_regression:
	def __init__(self):
		self.weights=None

	def fit(self,X,y):
		# Building Xbar 
		one = np.ones((X.shape[0], 1))
		Xbar = np.concatenate((one, X), axis = 1)
		# Calculating weights of the fitting line 
		A = np.dot(Xbar.T, Xbar)
		b = np.dot(Xbar.T, y)
		self.weights = np.dot(np.linalg.pinv(A), b)
		return self
	def predict(self):
		x0 = np.linspace(min(X0), max(X0) , 10 )
		x1 = np.linspace(min(X1), max(X1) , 10 )
		x=np.array([x0,x1]).T
		one = np.ones((x.shape[0], 1))
		Xbar = np.concatenate((one,x), axis = 1)
		y0 = np.dot(Xbar, self.weights)
		return x0,x1,y0[:,0]


def plot_synthetic(model):
	print( ' Weights : ', model.weights )
	fig = plt.figure()
	x0,x1,y0 = model.predict()
	               # the fitting line
	ax = plt.axes(projection='3d')
	ax.plot3D(X0,X1,Y1,'ro')
	for x in range(len(x1)):
		ax.plot3D(x0,np.linspace(x1[x], x1[x] , 10 ),y0,'blue')
	ax.set_xlabel('Depthm')
	ax.set_ylabel('T_degC')
	ax.set_zlabel('Salnty')
	plt.show()
model=linear_regression()
model.fit(X,y)
plot_synthetic(model)


#dung thu vien 
# from sklearn import datasets, linear_model

# # fit the model by Linear Regression
# one = np.ones((X.shape[0], 1))
# Xbar = np.concatenate((one, X), axis = 1)
# regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# regr.fit(Xbar, y)

# print( 'Solution found by scikit-learn  : ', regr.coef_ )

# x0 = np.linspace(min(X0), max(X0) , 100 )
# x1 = np.linspace(min(X1), max(X1) , 100 )
# y0 = regr.coef_[0][0]+regr.coef_[0][1]*x0+regr.coef_[0][2]*x1
# plt.plot(x0_real,y_real,'ro')
# plt.plot(x1_real,y_real,'bo')
# plt.plot(x0, y0,'b-') 
# plt.plot(x1, y0,'r-')               # the fitting line
# plt.xlabel(parameters)
# plt.ylabel(objective)
# plt.show()