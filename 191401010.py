# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NeuralNetwork import *    

# %%
#read housing data
data = pd.read_csv('housing.csv')

data.describe()

# %%
data.info()

# %%
#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop('PRICE', axis=1), data['PRICE'], test_size=0.2, random_state=12345)

# %%
#split train data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=12345)

# %%
X_train.shape


# %%
X_train_np=np.array(X_train)
y_train_np=np.array(y_train)
X_train_nn=X_train_np.reshape(X_train.shape[0],1,13)
y_train_nn=y_train_np.reshape(y_train.shape[0],1,1)

X_val_np=np.array(X_val)
y_val_np=np.array(y_val)
X_val_nn=X_val_np.reshape(X_val.shape[0],1,13)
y_val_nn=y_val_np.reshape(y_val.shape[0],1,1)

X_test_np=np.array(X_test)
y_test_np=np.array(y_test)
X_test_nn=X_test_np.reshape(102,1,13)
y_test_nn=y_test_np.reshape(102,1,1)

# %%
#create the network with 0 hidden layers
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

#train the network
error_train_hid0, error_val_hid0 = network.fit(X_train_nn, y_train_nn, epochs=1000, learning_rate=0.000001, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
hid0_err=mse(y_val_nn, y_pred)
hid0_test_err=mse(y_test_nn, network.predict(X_test_nn))

# %%
#create the network with 1 hidden layers,
#with 10 neurons in the hidden layer
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

#train the network
error_train_hid1, error_val_hid1 = network.fit(X_train_nn, y_train_nn, epochs=1000, learning_rate=0.000001, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
hid1_err=mse(y_val_nn, y_pred)
hid1_test_err=mse(y_test_nn, network.predict(X_test_nn))

# %%
#create the network with 2 hidden layers,
#with 10 neurons in the hidden layer
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 64))
network.add(ActivationLayer(relu, relu_prime))
network.add(Dense(64, 64))
network.add(ActivationLayer(softmax, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

#train the network
error_train_hid2, error_val_hid2 = network.fit(X_train_nn, y_train_nn, epochs=1000, learning_rate=0.000005, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
hid2_err=mse(y_val_nn, y_pred)
hid2_test_err=mse(y_test_nn, network.predict(X_test_nn))


# %%
#print hid0,hid1 and hid2 test error
print('Test error for 0 hidden layers: ', hid0_test_err)
print('Test error for 1 hidden layers: ', hid1_test_err)
print('Test error for 2 hidden layers: ', hid2_test_err)

# %%
#plot the error vs epochs
plt.plot(error_train_hid0, label='train')
plt.plot(error_val_hid0, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for 0 hidden layers')
plt.show()

#plot the error vs epochs
plt.plot(error_train_hid1, label='train')
plt.plot(error_val_hid1, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for 1 hidden layer')
plt.show()

#plot the error vs epochs
plt.plot(error_train_hid2, label='train')
plt.plot(error_val_hid2, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for 2 hidden layers')
plt.show()

#merge all plots
plt.plot(error_train_hid0, label='train')
plt.plot(error_val_hid0, label='validation')
plt.plot(error_train_hid1, label='train')
plt.plot(error_val_hid1, label='validation')
plt.plot(error_train_hid2, label='train')
plt.plot(error_val_hid2, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for 0, 1 and 2 hidden layers')
plt.show()

# %%


# %%
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 64))
network.add(ActivationLayer(relu, relu_prime))
network.add(Dense(64, 64))
network.add(ActivationLayer(softmax, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

error_train_n0, error_val_n0 = network.fit(X_train_nn, y_train_nn, epochs=670, learning_rate=0.000007, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
n0_err=mse(y_val_nn, y_pred)
n0_test_err=mse(y_test_nn, network.predict(X_test_nn))

# %%
network = NeuralNetwork()
network.add(Dense(13, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 64))
network.add(ActivationLayer(relu, relu_prime))
network.add(Dense(64, 32))
network.add(ActivationLayer(softmax, tanh_prime))
network.add(Dense(32, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

error_train_n1, error_val_n1 = network.fit(X_train_nn, y_train_nn, epochs=670, learning_rate=0.000007, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
n1_err=mse(y_val_nn, y_pred)
n1_test_err=mse(y_test_nn, network.predict(X_test_nn))

# %%
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 128))
network.add(ActivationLayer(relu, relu_prime))
network.add(Dense(128, 64))
network.add(ActivationLayer(softmax, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

error_train_n2, error_val_n2 = network.fit(X_train_nn, y_train_nn, epochs=670, learning_rate=0.000007, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
n2_err=mse(y_val_nn, y_pred)
n2_test_err=mse(y_test_nn, network.predict(X_test_nn))

# %%
#print n0,n1 and n2 test error
print('Test error for constant neuron: ', n0_test_err)
print('Test error for decreasing neuron: ', n1_test_err)
print('Test error for increasing and decreasing neuron: ', n2_test_err)

# %%
#plot the error vs epochs
plt.plot(error_train_n0, label='train')
plt.plot(error_val_n0, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for constant neurons')
plt.show()

#plot the error vs epochs
plt.plot(error_train_n1, label='train')
plt.plot(error_val_n1, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for increasing neurons')
plt.show()

#plot the error vs epochs
plt.plot(error_train_n2, label='train')
plt.plot(error_val_n2, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for increasing and decreasing neurons')
plt.show()


#merge all plots
plt.plot(error_train_n0, label='train')
plt.plot(error_val_n0, label='validation')
plt.plot(error_train_n1, label='train')
plt.plot(error_val_n1, label='validation')
plt.plot(error_train_n2, label='train')
plt.plot(error_val_n2, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for different architectures')
plt.show()

# %%
import time

# %%
network = NeuralNetwork()
network.add(Dense(13, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)


#train the network
error_train_lr0, error_val_lr0 = network.fit(X_train_nn, y_train_nn, epochs=700, learning_rate=0.0000001, x_val=X_val_nn, y_val=y_val_nn)


#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
lr1_err=mse(y_val_nn, y_pred)

# %%
network = NeuralNetwork()
network.add(Dense(13, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)


#train the network
error_train_lr1, error_val_lr1 = network.fit(X_train_nn, y_train_nn, epochs=700, learning_rate=0.0000005, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
lr1_err=mse(y_val_nn, y_pred)

# %%
network = NeuralNetwork()
network.add(Dense(13, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

#train the network

error_train_lr2, error_val_lr2 = network.fit(X_train_nn, y_train_nn, epochs=700, learning_rate=0.000001, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
lr2_err=mse(y_val_nn, y_pred)

# %%


# %%
#plot the error vs epochs
#show all error for different learning rates on same plot
plt.plot(error_train_lr0, label='train_lr0')
plt.plot(error_val_lr0, label='validation_lr0')
plt.plot(error_train_lr1, label='train_lr1')
plt.plot(error_val_lr1, label='validation_lr1')
plt.plot(error_train_lr2, label='train_lr2')
plt.plot(error_val_lr2, label='validation_lr2')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for different learning rates (same train and validation data)')
plt.show()


# %%
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 256))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(256, 128))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(128, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)


#train the network
error_train_epoch, error_val_epoch = network.fit(X_train_nn, y_train_nn, epochs=3000, learning_rate=0.0000001, x_val=X_val_nn, y_val=y_val_nn)

#predict the output
y_pred = network.predict(X_val_nn)

#calculate the mean squared error
epoch_err=mse(y_val_nn, y_pred)

# %%
#plot error vs epochs
plt.plot(error_train_epoch, label='train')
plt.plot(error_val_epoch, label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for increasing epochs')

# %%
best_network = NeuralNetwork()
best_network.add(Dense(13, 128))
best_network.add(ActivationLayer(sigmoid, sigmoid_prime))
best_network.add(Dense(128, 32))
best_network.add(ActivationLayer(relu, relu_prime))
best_network.add(Dense(32, 1))
best_network.add(ActivationLayer(relu, relu_prime))
best_network.use(mse, mse_prime)


#train the network
error_train_best, error_val_best = best_network.fit(X_train_nn, y_train_nn, epochs=1000, learning_rate=0.000001, x_val=X_val_nn, y_val=y_val_nn)


#calculate test error
y_pred_test = best_network.predict(X_test_nn)
test_err=mse(y_test_nn, y_pred_test)
test_err

# %%



