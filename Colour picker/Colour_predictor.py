from __future__ import division
import numpy as np
from training_data import *

#Creates a neural network class
class neural_network():

	#Initialisation takes a single input.
	def __init__(self, rand_W):
		np.random.seed(1)

		# If input is 1, random weights are chosen to start and the network can then be trained within the UI.
		if (rand_W == 1):
			self._IH_weights = 2 * np.random.random((3,4))-1
			self._HO_weights = 2 * np.random.random((1,4))-1
		# Otherwise, pretrained weights are used (found in training_data.py).
		else:
			self._IH_weights = np.asmatrix(starting_IH_weights)
			self._HO_weights = np.asmatrix(starting_HO_weights)

	#Sigmoid activation function definition (for feed forward).
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	#Sigmoid derivative function definition (for back propagation).
	def sigmoid_derivative(self, x):
		return np.multiply (x , (1 - x))    
    
	#Feed forward algorithm.
	def get_output(self, _input):
		#Input transposed and normalised.
		T_input = np.transpose(np.asmatrix(_input* (1/255))) 
		#Bias input added to transposed input.
		T_input = np.vstack([T_input,1])
		#Feedforward equation implemented for hidden layers.
		hidden = self.sigmoid(np.dot(self._IH_weights, T_input))
		#Bias input added to hidden outputs.
		hidden = np.vstack([hidden,1])
		#Feedforward equation implemented for single output.
		output = self.sigmoid(np.dot(self._HO_weights, hidden))
		return hidden, output

	#Back propagation algorithm in order to train the network. 
	def back_propagation(self, training_inputs, learning_rate, lr_decay_rate, iterations):
		lr = learning_rate
		print_counter = 0
		
		#Iterates through training algorithm 'iterations' number of times
		for iteration in xrange(iterations):
			#Chooses a random colour (black, white or grey)
			rand_col = np.random.random_integers(0,2)
			#Chooses a random input depnding on the colour chosen and the number of inputs for that colour in training data.
			inp_num = len(training_inputs[rand_col])-1
			rand_inp = np.random.random_integers(0,inp_num)

			#Calculates hidden and output layer outputs through feed forward.
			hidden = self.get_output(training_inputs[rand_col][rand_inp])[0]
			output = self.get_output(training_inputs[rand_col][rand_inp])[1]

			#Transposed, normalised and bias input added.
			T_input = np.vstack([np.transpose(np.asmatrix(training_inputs[rand_col][rand_inp])) * (1/255),1])
			#Input in usable 'untransposed' form
			_input = np.transpose(np.asmatrix(T_input))
			#Transposed hidden layer outputs
			T_hidden = np.transpose(hidden)

			#Maps colour chosed to a 0-1 value.
			training_target = (rand_col)/2

			#Calculation of hidden and output errors
			_O_error = training_target - output
			T_HO_weights = np.transpose(self._HO_weights)
			_H_error = T_HO_weights * _O_error

			#Calculation of change in weights required depending in the errors
			delta_HO = np.multiply(lr * (np.multiply(_O_error, self.sigmoid_derivative(output))), T_hidden)
			delta_IH = np.multiply(lr * (np.multiply(_H_error, self.sigmoid_derivative(hidden))), _input)
			delta_IH = np.delete(delta_IH, 3, 0)

			#Weights updated 
			self._HO_weights = self._HO_weights + delta_HO
			self._IH_weights = self._IH_weights + delta_IH

			#Learning rate updated
			lr = lr * lr_decay_rate

			#Prints useful information every 1000 iterations
			if (print_counter == 1000):
				print training_target, rand_inp
				print _O_error
				print ""
				print lr
				print ""
				print iteration
				print_counter = 0
			print_counter += 1

	

