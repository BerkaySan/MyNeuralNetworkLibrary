This library serves as a Neural Network implementation.
It is straightforward to use and was created in Python.
Regression or classification may be done using it.
Unfortunately, batch size and alternative loss function is not supported by the library. Default loss function is MSE and default batch size 1. 
For huge datasets, it is not advised to utilize it. 

Using of the library is very simple: 
1. Create a NeuralNetwork object
2. Create a Dense object for each layer in the network
Parameters: 
    - Input dimension
    - Output dimension
3. Create a Activation object for each layer in the network
Parameters:
    - - Activation function (sigmoid, softmax, tanh, relu)
4. Add the Dense and Activation objects to the NeuralNetwork object
4. Compile the NeuralNetwork object
5. Train the NeuralNetwork object
Parameters:
    - Training data
    - Training labels
    - Number of epochs
    - Learning rate
6. Predict using the NeuralNetwork object

Other methods:
- use - used the loss function in the model
- fit - trains the network
- predict - predicts the output



Example of the using library:
from NeuralNetwork import NeuralNetwork
network = NeuralNetwork()
network.add(Dense(13, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 64))
network.add(ActivationLayer(tanh, tanh_prime))
network.add(Dense(64, 1))
network.add(ActivationLayer(relu, relu_prime))
network.use(mse, mse_prime)

#train the network
error_train, error_val = network.fit(X_train, y_train, epochs=1000, learning_rate=0.000001, x_val=X_val, y_val=y_val)

#predict the output
y_pred = network.predict(X_test)