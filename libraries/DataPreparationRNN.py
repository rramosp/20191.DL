import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
#split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def PintaResultado(dataset,trainPredict,testPredict,look_back):
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(dataset)-len(testPredict)-1:len(dataset)-1, :] = testPredict
	# plot baseline and predictions
	plt.plot(dataset)
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

def EstimaRMSE(model,X_train,X_test,y_train,y_test,scaler,look_back):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict

def EstimaRMSE_RNN(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],n_steps,look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],n_steps,look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict