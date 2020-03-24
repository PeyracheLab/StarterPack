#!/usr/bin/env python

'''
	File name: main_exercise3.py
	Author: Guillaume Viejo
	Date created: 23/03/2020
	Python Version: 3.6

Machine learning 101
'''

###############################################################################
# 1. Import numpy, matplotlib and pandas
###############################################################################


###############################################################################
# 2. Some data 
###############################################################################
traindata = np.vstack([
				np.random.multivariate_normal([1,1], np.eye(2)*0.1, 100),
				np.random.multivariate_normal([-1,-1], np.eye(2)*0.1, 100),
				])

testdata = np.vstack([
				np.random.multivariate_normal([1,1], np.eye(2)*1, 100),
				np.random.multivariate_normal([-1,-1], np.eye(2)*1, 100),
				])

labels = np.hstack([np.zeros(100), np.ones(100)])

###############################################################################
# 3. Some parameters
###############################################################################
alpha = 0.01

###############################################################################
# 3. Define a list variable called errors
###############################################################################


###############################################################################
# 4. Define an array called weights of size 2 using the np.random.randn function
###############################################################################



###############################################################################
# 5. Define a float variable called bias using the np.random.randn function
###############################################################################



###############################################################################
# 6. start a loop iterating over i for 100000 iterations
###############################################################################


	###############################################################################
	# 7. Define an integer variable called idx randomly sampled between 0 and the 
	# size of traindata (in the first dimension of the array)
	###############################################################################	
	

	###############################################################################
	# 8. Declare the variable inputs as the element of traindata located in the position defined 
	# by idx (inputs should be an array of size 2)
	# Try it first in your terminal if you are not sure
	###############################################################################	
	


	###############################################################################
	# 9. Declare the variable activation as the dot product (hint : np.dot) between 
	# weights and inputs. Add the variable bias to the variable activation
	###############################################################################	
	


	###############################################################################
	# 10. Compute the sigmoid function of the variable activation and name the output
	# prediction (https://en.wikipedia.org/wiki/Sigmoid_function)
	###############################################################################		
	


	###############################################################################
	# 11. Compute the variable error as the difference between the value in the variable 
	# labels at the position idx and the variable prediction. So error is a float value
	###############################################################################		
	

	###############################################################################
	# 12. Assign weights as the weights plus alpha * error * traindata[idx]. Easy
	###############################################################################		
	

	###############################################################################
	# 13. Assign bias as bias + alpha * error
	###############################################################################	
	

	###############################################################################
	# 14. Append to the list errors the absolute value of the variable error
	###############################################################################	
	


###############################################################################
# 15. End of the loop. It's time to see if your neural netwrok has learned
# Repeat steps 9 and 10 but this time replacing traindata with testdata
# Easiest way is to make a loop iterating ONLY ONCE over each element of testdata
# Make sure you store the variable prediction in an array for each testdata element
# You can define a list called testprediction for example and append each prediction
###############################################################################	



###############################################################################
# 16. Make a 1 by 3 figures with matplotlib. Subplot 1 should display the variable errors
# Subplot 2 should display a 2d scatter plot of traindata points. Points should be colored
# based on the value in labels
# Subplot 3 should display a 2d scatter plot of testdata points. Points should be colored
# based on the value in test prediction
###############################################################################	



###############################################################################
# 16. If your code is correct, the final figure should  look like this :
# https://www.dropbox.com/s/6n4ai6w5l19o4ss/figure_exercise3.png?dl=1
# You have coded a simple linear perceptron
# Now there is one parameter that is missing here that would lower the error.
# Can you find it to make the error look like this :
# https://www.dropbox.com/s/j4nzizasg00izdt/figure_exercise3_2.png?dl=1
###############################################################################	


