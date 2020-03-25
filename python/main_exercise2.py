#!/usr/bin/env python

'''
	File name: main_exercise1.py
	Author: Guillaume Viejo
	Date created: 23/03/2020
	Python Version: 3.6

A bit more complex
'''

###############################################################################
# 1. Import numpy, matplotlib and pandas
###############################################################################


###############################################################################
# 2. Some parameters
###############################################################################
r = 0.3 # radius
phi = 0.0 # angle
omega = 3.4 # vitesse angulaire
sigma  = -0.8 # parametre
h = 0.001


###############################################################################
# 3 Define an integer n equal to 40000
###############################################################################


###############################################################################
# 4 Define the variable called cosphi as r times cosinus of phi
# and the variable sinphi as r times sinus of phi
###############################################################################


###############################################################################
# 5 Use the variable n to define an array of zero of size (n, 5) called data
###############################################################################



###############################################################################
# 6 Now make a loop iterating over n and define your iterator as i
# Remember : everything inside the loop should be tabulated in regard for the for line.
# To help you, the next steps are tabulated
###############################################################################


	
	###############################################################################
	# 7 Start with a condition : if i larger than 10000, assign 0.99 to sigma
	###############################################################################
	


	###############################################################################
	# 8 Second condition : if i larger than 30000, assign -0.99 to sigma
	###############################################################################
	


	###############################################################################
	# 9 Now to help you, the next line of code is already written 
	###############################################################################
	r = h*(np.tanh(sigma)*r - r**3.0) + r

	###############################################################################
	# 10 Assign to the variable phi the following value phi = h * omega + phi (yeah easy)
	###############################################################################
	

	###############################################################################
	# 11 Compute modulo 2*np.pi of phi and assign it to phi
	# Hint : a%=(1) is a correct sentence in python
	###############################################################################
	

	###############################################################################
	# 12 Assign to the variables cosphi and sinphi the new value r times cosinus of phi
	# and r times sinus of phi
	###############################################################################


	###############################################################################
	# 13 Using the iterator i, assign at the ith posiiton in the first column of data the value r
	# Remember in python, array indexing start at 0
	###############################################################################

	
	###############################################################################
	# 14 same for phi, cosphi, sinphi and sigma for columns 2, 3, 4 and 5 of data
	###############################################################################



###############################################################################
# 15 End of the loop. Now is time to plot the data. So make sure your code now is 
# outside the loop. 
# Make a two by two grid in matplotlib.
# The top left plot should display the hyperbolic tangent of the sigma variable (in 
# the last column of data)
# The top right column should display the r variable (first column)
# Bottom left should display cosphi and sinphi together. Try to put a legend to
# differentiate the two
# Bottom right should display cosphi versus sinphi
###############################################################################



###############################################################################
# 16 If you code is correct by running the entire script at once, 
# the figure should look like this :
# https://www.dropbox.com/s/c58an08h235ztcj/figure_exercise2.png?dl=1
# Congrats, you have simulated a Hopf bifurcation :
# https://en.wikipedia.org/wiki/Hopf_bifurcation
###############################################################################

