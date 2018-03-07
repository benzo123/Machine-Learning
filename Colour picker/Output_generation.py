from __future__ import division
import numpy as np
from training_data import *
from Colour_predictor import *
from graphics import *


def main():

	np.random.seed()
	
	#Creates a new neural network instance
	output_network = neural_network(1)

	#Creates UI window
	win = GraphWin('Random tests', 1000, 500)
	text1 = Text(Point(win.getWidth()/2, win.getHeight()/2), 'White or Black?')
	text1.setSize(30)
	text1.setStyle("bold")
	text1.draw(win)
	text2 = Text(Point(win.getWidth()/2, win.getHeight()/2 + 100), 'Train')
	text2.setSize(15)
	text2.setStyle("normal")
	text2.draw(win)
	top_left = Point(win.getWidth()/2-30,win.getHeight()/2+100 - 20)
	botton_right = Point(win.getWidth()/2+30,win.getHeight()/2+100 + 20)
	train_box = Rectangle(top_left,botton_right)
	train_box.draw(win)

	while True:
		#Generates a random colour to test the network and sets the background as that colour
		test_colour = np.random.randint(256,size=3)
		win.setBackground(color_rgb(test_colour[0],test_colour[1], test_colour[2]))
		
		#Gets the network output
		prob = output_network.get_output(test_colour)[1]
		print test_colour
		print prob

		click_X = 0
		click_Y = 0
		
		#If prob is less than 0.5, colour is black.
		if (prob < 0.5):
			text1.setTextColor('Black')
			text2.setTextColor('Black')
			train_box.setOutline('Black')
		#If prob is greater than or equal to 0.5, colour is white
		if (prob >= 0.5):
			text1.setTextColor('White')
			text2.setTextColor('White')
			train_box.setOutline('White')

		#uses the get mouse function to attain where and when mouse is clicked
		clickPoint = win.getMouse()
		print clickPoint
		click_X = clickPoint.getX()
		click_Y = clickPoint.getY()

		#If click is within a certain bounding box, train the network for 100,000 iterations.
		if (click_X < win.getWidth()/2+30 and click_X > win.getWidth()/2-30 and click_Y < win.getHeight()/2+100 + 20 and click_Y > win.getHeight()/2+100 - 20):
			text2.setTextColor('blue')
			print ""
			print "starting _IH_weights"		
			print output_network._IH_weights
			print "random starting _HO_weights"		
			print output_network._HO_weights
			output_network.back_propagation(training_set, 0.005, 0.999999999, 100000)
			print ""
			print "new _IH_weights:"
			print output_network._IH_weights
			print "new _HO_weights:"
			print output_network._HO_weights
			print ""
		
main()