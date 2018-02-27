#Numpy used for the matrix maths. 
import numpy as np
#Creation of the neural network class.

class neural_network():
	#Initialisation involves the creation of random weight matrixes. 

    def __init__(self):
        np.random.seed(1)
        self._IH_weights = 2 * np.random.random((2,3))-1
        self._HO_weights = 2 * np.random.random((1,3))-1

    #Sigmoid activation function definition (for feed forward).
    def sigmoid(self, x):		 	
        return 1 / (1 + np.exp(-x))

    #Sigmoid derivative function definition (for backpropagation).
    def sigmoid_derivative(self, x):
        return np.multiply (x , (1 - x))    

    #Feed forward algorithm definition, for use in both the hidden and output layers.
    def feed_forward(self, inputs, weights):
        return self.sigmoid(np.dot(inputs, weights))

    #Performs the feedforward (including adding the bias) where [0] is the hidden output and [1] is the final output   
    def get_output(self, _input):
        hidden = self.feed_forward(self._IH_weights, np.transpose(np.asmatrix(_input))) 
        hidden = np.vstack([hidden,1])  
        output = self.feed_forward(self._HO_weights, hidden)
        return hidden, output

    #Performs the backpropagation algorithm with the inputs and targets, at a set learning rate for n iterations. 
    def back_propagation(self, training_input, training_target, lr, iterations):
    	#Iterates the backpropagation algorithm 'iterations' times.
        for iteration in xrange(iterations):
        	#Finds a random input (rather than cycling through all the inputs) to improve training speed
            rand_inp = np.random.random_integers(0,3)
            output = self.get_output(training_input[rand_inp])[1]
            hidden = self.get_output(training_input[rand_inp])[0]
            _input = np.asmatrix(training_input[rand_inp])

            #Transposed matricies of the input and hidden outputs.
            input_T = np.transpose(_input)
            hidden_T = np.transpose(hidden)

            #Calculation of output and hidden layer errors.
            _O_error = training_target[rand_inp] - output
    	    _HO_weights_T = np.transpose(self._HO_weights)
    	    _H_error = _HO_weights_T * _O_error

    	    #Calculation and addition of changes in weight matrcicies.
    	    delta_HO = np.multiply(lr * (np.multiply(_O_error, self.sigmoid_derivative(output))), hidden_T)
            delta_IH = np.multiply(lr * (np.multiply(_H_error, self.sigmoid_derivative(hidden))), _input)
            delta_IH = np.delete(delta_IH, 2, 0)
            self._HO_weights = self._HO_weights + delta_HO
            self._IH_weights = self._IH_weights + delta_IH

            #Prints Weight changes to visulaise decrease in error (optional).
            #print delta_IH, delta_HO

    	#return _H_error, _O_error, _HO_weights_T, delta_HO, delta_IH, _input
    

if __name__ == "__main__":
    #Intialise the neural network.
    neural_network = neural_network()
    #Training set inputs (can be set as anything provided the final input is always 1).
    training_inputs = np.array([[0,0,1],
                                [0,1,1],
                                [1,1,1],
                                [1,0,1]])
    #training set targets (can be set as anything).
    training_targets = np.array([[1],
                                 [1],
                                 [0],
                                 [0]])
    #Learning rate of the network [could add a decay over time function]
    lr = 0.05
    #Prints the starting weights.
    print "Random starting I to H weights: "
    print neural_network._IH_weights
    print "Random starting H to O weights: "
    print neural_network._HO_weights
    #Prints a blank line (seperates the trained from the untrained data).
    print ""
    #Training the neural network n times.
    neural_network.back_propagation(training_inputs,training_targets, lr, 50000)
    #Prints weights post training
    print "new I to H weights after training: "
    print neural_network._IH_weights
    print "new H to O weights after training: "
    print neural_network._HO_weights
	#Prints a blank line (seperates the trained data from the output tests).
    print ""
    #Prints the output tests.
    print "output test: ", np.transpose(training_targets)
    print neural_network.get_output([0,0,1])[1]
    print neural_network.get_output([0,1,1])[1]
    print neural_network.get_output([1,1,1])[1]
    print neural_network.get_output([1,0,1])[1]


"""
	   _____
	  /	 .. \ 
 ____/_______\____
/_o_o_o_o_o_o_o_o_\  		| ======= In the coldest night ======= |
\_o_o_o_|_|_o_o_o_/			| === They come from a falling sky === |
	    / \					| ====== Shining brilliant light ===== |
	   /   \
	  /     \
	 / _O/   \
	/    \    \      
   /	 /\_   \     
  / 	 \  ` 	\
 /	     `   	 \ 

     `    
"""



