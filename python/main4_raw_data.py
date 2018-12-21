#!/usr/bin/env python

'''
	File name: main4_raw_data.py
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

This scripts will show you how to use the wrappers function to load raw data
A typical preprocessing pipeline shall output 
	- Mouse-Session.clu.*
	- Mouse-Session.res.*
	- Mouse-Session.fet.*
	- Mouse-Session.spk.*
	- Mouse-Session.xml
	- Mouse-Session.eeg
	- Epoch_TS.csv
	- Tracking_data.csv



The data should be found in StarterPack/data_raw/A1110-180621/

This script will show you how to load the various data you need

The function are already written in the file wrappers.py that should be in the same directory as this script

To speed up loading of the data, a folder called /Analysis will be created and some data will be saved here
So that next time, you load the script, the wrappers will search in /Analysis to load faster
'''

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *

# first we define a string for the data directory
# It is usually better to separate the data from the code
data_directory = '../data_raw/A1110-180621'
# The two dots means we go backward in the folder architecture and then into the data folder

# To list the files in the directory, you use the os package (for Operating System) and the listdir function
import os
files = os.listdir(data_directory) 
# Check your variables by typing files in your terminal
files

# First thing to load are the spikes.
# Here you can use the wrappers loadSpikeData
from wrappers import loadSpikeData
spikes, shank = loadSpikeData(data_directory)
# Type your variables in the terminal to see what it looks like

# Second thing is some information about the recording session like the geometry of the shanks and sampling frequency
# You can use the loadXML wrapper
from wrappers import loadXML
n_channels, fs, shank_to_channel = loadXML(data_directory)
# Again type your variables

# Third thing is to load the epoch of wake because you want to compute a tuning curve.
# One thing to know is what was the structure of the recording day
# In this case, sleep alternate with wake
from wrappers import makeEpochs, loadEpoch
makeEpochs(data_directory, ['sleep', 'wake', 'sleep', 'wake'], file='Epoch_TS.csv')
wake_ep = loadEpoch(data_directory, 'wake')
# Check the folder /Analysis/ in data_directory to see the file BehavEpoch that should have appeared

# Last thing is the positon of the animal 
from wrappers import loadPosition
position = loadPosition(data_directory, file = 'Tracking_data.csv', ep = wake_ep)

# We can look at the position of the animal in 2d with a figure
figure()
plot(position['x'], position['y'])
show()



# Now we are going to compute the tuning curve for all neurons during exploration
# The process of making a tuning curve has been covered in main3_tuningcurves.py
# So here we are gonna define a function that will be looped over each ADn neurons

def computeAngularTuningCurve(spike_ts, angle_tsd, nb_bins = 60, frequency = 120.0):
	angle_spike = angle_tsd.realign(spike_ts)
	bins = np.linspace(0, 2*np.pi, nb_bins)
	spike_count, bin_edges = np.histogram(angle_spike, bins)
	occupancy, _ = np.histogram(angle, bins)
	spike_count = spike_count/occupancy
	tuning_curve = spike_count*frequency
	tuning_curve = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = tuning_curve)
	return tuning_curve

# Let's prepare a dataframe to receive our tuning curves
id_animal = 'A1110-180621'
column_names = [id_animal+'_'+str(k) for k in spikes.keys()]
tuning_curves = pd.DataFrame(columns = column_names)

# let's do the loop
for k in spikes.keys():
	spks = spikes[k]
	angle = position['ry']
	# first thing is to restrict the data to the exploration period
	spks = spks.restrict(wake_ep)
	# second we can call the function
	tcurve = computeAngularTuningCurve(spks, angle)
	# third we can add the new tuning curve to the dataframe ready
	tuning_curves[id_animal+'_'+str(k)] = tcurve
	
# And let's plot all the tuning curves

from pylab import *
figure()
plot(tuning_curves)
xlabel("Head-direction (rad)")
ylabel("Firing rate")
title("ADn neuron")
grid()
show()

# Even cooler we can do a polar plot
figure()
subplot(111, projection='polar')
plot(tuning_curves)
xlabel("Head-direction (rad)")
ylabel("Firing rate")
title("ADn neuron")
grid()
show()


