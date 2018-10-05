#!/usr/bin/env python

'''
	File name: main4.py
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

The goal of this script is to plot the tuning curve of one neuron of ADn
The steps are
1) load the data
2) integrate the data in pandas frame which is a data helper
3) construct the angular bins
4) compute the mean firing rate for each bin
5) plot

To start, I use the same function as in main3.py to load the spiking activity
'''
import numpy as np
import pandas as pd
import neuroseries as nts
import os
from functions import *
import scipy.io
data_directory 	= '../data/'

files 			= os.listdir(data_directory) 
generalinfo 	= scipy.io.loadmat(data_directory+'Mouse12-120806_GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
spikes,shank 	= loadSpikeData(data_directory+'Mouse12-120806_SpikeData.mat', shankStructure['thalamus'])
my_thalamus_neuron_index = list(spikes.keys())
hd_neuron_index = loadHDCellInfo(data_directory+'Mouse12-120806_HDCells.mat', my_thalamus_neuron_index)
hd_spikes = {}
for neuron in hd_neuron_index:
	hd_spikes[neuron] = spikes[neuron]

wake_ep 		= loadEpoch(data_directory, 'wake')
sleep_ep 		= loadEpoch(data_directory, 'sleep')
sws_ep 			= loadEpoch(data_directory, 'sws')
rem_ep 			= loadEpoch(data_directory, 'rem')

# Next step is to load the angular value at each time steps
# We need to load Mouse12-120806_PosHD.txt which is a text file
# We can use the function genfromtxt of numpy that load a simple text file
data = np.genfromtxt(data_directory+'Mouse12-120806_PosHD.txt')
# Check your variable by typing 
data
# It's an array, we can check the dimension by typing :
data.shape
# It has 40858 lines and 4 columns
# The columns are respectively [times | x position in the arena | y position in the arena | angular value of the head]
# So we can use the TsdFrame object of neuroseries as seen in main2.py
mouse_position = nts.TsdFrame(d = data[:,[1,2,3]], t = data[:,0], time_units = 's')
# Check your variable
mouse_position
# By defaut, TsdFrame does not take column name as input 
mouse_position.columns
# But TsdFrame is a wrapper of pandas and you can change name of columns in pandas
# So let's change the columns name
mouse_position.columns = ['x', 'y', 'ang']

# It's good to always check the data by plotting them
# To see the position of the mouse in the arena during the session
# you can plot the x position versus the y position
import matplotlib.pyplot as plt
plt.figure()
plt.plot(mouse_position['x'].values, mouse_position['y'].values)
plt.xlabel("x position (cm)")
plt.ylabel("y position (cm)")
plt.show()

# To compute the tuning curve of one neuron, we need to restrict it's activity to epoch of active exploration
# epoch means we need to use the object nts.IntervalSet
# The interval are contained in the file : Mouse12-120806_ArenaEpoch.txt
new_data = np.genfromtxt(data_directory+'Mouse12-120806_ArenaEpoch.txt')
# Check your variables and look at it shape
# It's a 2 dimension table with start and end of epoch. We can integrate it in nts:
exploration = nts.IntervalSet(start = new_data[:,0], end = new_data[:,1], time_units = 's')
# Check your variable again for example in second:
exploration.as_units('s')


# Next step is to compute an average firing rate for one neuron
# Let's take neuron 39 (it's the neuron with the highest number of spikes)
my_neuron = spikes[39]
# To speed up computation, we can restrict the time of spikes to epoch of wake
my_neuron = my_neuron.restrict(wake_ep)

# A firing rate is the number of spikes per time bin
# We need to loop over each bin and find the corresponding number of spikes that falls in the interval
# So first you define the size of your time bin, for example let's take one second time bin
bin_size = 1.0 # second
# Next we need to know the duration of our recording 
# So we need the time of the first spike and the time of the last spike
first_spike = my_neuron.as_units('s').index[0]
last_spike = my_neuron.as_units('s').index[-1]
# Observe the -1 for the value at the end of an array
duration = last_spike - first_spike
# it's the time of the last spike
# To have no error with borders it's better to add 1 second 
duration = duration + 1.0
# this is duration in s, therefore it corresponds in minutes to 
print(duration/60) 
# it's the number of minute of wake of the animal
# with a bin size of 1 second, the number of points is 
nb_points = duration/bin_size
# which is equal to duration in fact (different for a different bin size...)
# let's convert nb_points to int to initiate the array (otherwise numpy will throw you an error)
nb_points = int(nb_points)
firing_rate = np.zeros(nb_points)

# Here is the most plain way to do it
# If you feel experienced, check the np.digitize function in google
# We will loop over each time position starting at the first spike and ending at the last spike

# But first we do it for the first position
# Starting at i = 0
i = 0
# the start of my first interval is basically i*bin_size + first_spike
start = i*bin_size + first_spike
# the end of my interval is the start value plus the size of my bin
end = start + bin_size
# We have an interval of [start, end] 
# Now we can search for the spikes that falls in this interval
# DONT FORGET THE TIME CONVERSION IN SECOND IN YOUR NEUROSERIE
spikes_in_interval = my_neuron.as_units('s').loc[start:end]
# Then firing rate is counting spikes so
number_of_spikes = len(spikes_in_interval)
# We can write that in the first position of the firing_rate array
firing_rate[i] = number_of_spikes
# Check your firing rate variable to see that it appeared in the first element

# Now the full loop will be this :
for i in range(nb_points):
	start = i*bin_size + first_spike
	end = start + bin_size
	spikes_in_interval = my_neuron.as_units('s').loc[start:end]
	number_of_spikes = len(spikes_in_interval)
	firing_rate[i] = number_of_spikes

# It can be very long because it's not the optimal way of doing it
# When it's done, check your variable firing_rate
# We can put firing rate in a nts.Tsd
firing_rate = nts.Tsd(t = np.arange(first_spike, last_spike, bin_size), d = firing_rate, time_units = 's')

# Next step is to compute the average angular direction with the same time bin
# Steps are the same
head_direction = np.zeros(nb_points)
for i in range(nb_points):	
	start = i*bin_size + first_spike
	end = start + bin_size
	head_direction_in_interval = mouse_position['ang'].as_units('s').loc[start:end]
	average_head_direction = np.mean(head_direction_in_interval)
	head_direction[i] = average_head_direction

head_direction = nts.Tsd(t = np.arange(first_spike, last_spike, bin_size), d = head_direction, time_units = 's')

# we can put the two array together
my_data = np.vstack([firing_rate.values, head_direction.values]).transpose()
# and put them in panda 
my_data = nts.TsdFrame(t = np.arange(first_spike, last_spike, bin_size), d = my_data)
# and give name to column
my_data.columns = ['firing', 'angle']
# Check you variable and observe the NaN value in the angle column
# NaN means not a number
# We want to get rid of them so you can call the isnull() function of pandas
# Check the following call 
my_data.isnull()
# Observe the False and True value
# We want the time position that is the inverse of False when calling isnull
# So you call the angle column
my_data.isnull()['angle']
# And you invert the boolean with ~
~my_data.isnull()['angle'].values
# now we can downsample my_data with index that are only True in the previous line
my_data = my_data[~my_data.isnull()['angle'].values]
# Check your variable my_data to see that the NaN have disappeared

# Last step is to compute the tuning curve over 60 points between 0 and 2pi
tuning_curve = np.zeros(60)
# the tuning curve is the mean firing rate per angular bins
# First step is to define the angular bins
angular_bins = np.linspace(0, 2*np.pi, 61)
# Check your variable; it goes from 0 to 2pi plus a small value for border effect
# It's the border of your bins, therefore you will have 61 - 1 = 60 points
# we can loop trough the angular bins and search for points in my_data that falls in this angular interval
# again we can do the first bins outside the loop
left_border = angular_bins[0]
right_border = angular_bins[1]
# We can use the great function np.logical_and here and ask for the point
# that are greater than left_border and points that are lesser than right_border
index = np.logical_and(my_data['angle']>=left_border, my_data['angle']<right_border).values
# Check your index variable filled with False
# But somewhere lies some True 
# we can thus index my_data with this variable
first_bin = my_data[index]
# Check your variable to see that the value at the angle falls between the border of your bin

# Now we can loop
for i in range(60):
	left_border = angular_bins[i]
	right_border = angular_bins[i+1]
	index = np.logical_and(my_data['angle']>left_border, my_data['angle']<=right_border).values	
	tuning_curve[i] = np.mean(my_data[index]['firing'])

# We can plot it

plt.figure()
phase = np.linspace(0, 2*np.pi, 60)
plt.plot(phase, tuning_curve)
plt.xlabel("Head-direction (rad)")
plt.ylabel("Firing rate")
plt.title("ADn neuron")
plt.grid()
plt.show()

# Even cooler we can do a polar plot
plt.figure()
plt.subplot(111, projection='polar')
plt.plot(phase, tuning_curve)
plt.xlabel("Head-direction (rad)")
plt.ylabel("Firing rate")
plt.title("ADn neuron")
plt.grid()
plt.show()

# Exercice :
# 1 second bin is a poor resolution
# You can try to do the tuning curve again with smaller time bins
# You can altough restrict the spike to the intervals of active exploration by using the restrict function
# Last you can write a loop to compute the tuning curve for each neuron
# Hint : compute the firing rate for all neurons at the same time and put it in a TsdFrame, it will then be 
# easier to restrict all neurons at once to one epoch.
# You can try to do a raster plot as well i.e. time of spike versus neurons


