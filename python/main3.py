#!/usr/bin/env python

'''
	File name: main3.py
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

This script will show you how to load the various data you need
The function are already written in the file functions.py that should be in the same directory
The data should be found in StarterPack/data/
'''

import numpy as np
import pandas as pd
import neuroseries as nts

# first we define a string for the data directory
# It is usually better to separate the data from the code
data_directory = '../data/'
# The two dots means we go backward in the folder architecture and then into the data folder

# To list the files in the directory, you use the os package (for operating system) and the listdir function
import os
files = os.listdir(data_directory) 
# Check your variables by typing files in your terminal
files

# All files that have a .mat extension are data that have been saved with Matlab
# At some point, data files should be saved to be easily compatible for matlab and python
# But those are not
# Except for the file called MouseID_generalInfo.mat, you should use the Wrappers provided in functions.py

# First we load general info about the session recording
# We need the scipy library which is another scientific library and the io.loadmat function
import scipy.io
generalinfo = scipy.io.loadmat(data_directory+'Mouse12-120806_GeneralInfo.mat')
# Type your variable in your terminal and observe the results
generalinfo
# The type of your variable is a dictionnary
print(type(generalinfo))
# Therefore you can see the keys of your dict by typing 
generalinfo.keys()
# What is interesting here is the shankStructure
# The recording probe is made of shanks (usually 8)
# And shanks can go in different regions of the brain
# If you type :
generalinfo['shankStructure']
# you see that it return a matlab object with name of region i.e. hippocampus, thalamus, pfc ...
# To parse this object is complex, therefore you should use now the wrapper I provided in functions.py
from functions import loadShankStructure
# You call the function by giving the generalinfo dictionnary
shankStructure = loadShankStructure(generalinfo)
# You check your variable by typing :
shankStructure
# And now each region is associated with a list of number.
# Each number indicates the index of a shank
# For example, the thalamus shanks index are :
shankStructure['thalamus']
# Therefore we have here 8 shanks in the thalamus
# This will be useful to know which spikes were recorded in the thalamus

# Now we can load the spikes in Mouse12-120806_SpikeData.mat by using another wrapper
from functions import loadSpikeData
# and we want only the spikes from the thalamus
# So you need to pass the shankStructure of the thalamus as an argument of the function 
spikes,shank = loadSpikeData(data_directory+'Mouse12-120806_SpikeData.mat', shankStructure['thalamus'])
# It returns you two things:
# you can look at them by typing them in your terminal
spikes
shank
# and learn their types by using
type(spikes)
type(shank)
# spikes is a dictionnary with index of one neuron as key and the time series of spike as item
# To see the keys :
spikes.keys()
# There should be values ranging from 0 to 40 therefore we have 41 neurons in this recording
# To acces one neuron:
spikes[0]
spikes[4]
spikes[24]
# It returns a Ts object with the time occurence of spikes on the left column and NaN in the right columns
# the variable shank is an array of the total number of neurons recorded in this session trough all brain areas with the shank index
# But it is not important here so you can forget about it

# Which neurons of the thalamus are head-direction neurons?
# To know that, you need to load Mouse12-120806_HDCells.mat
# But first, you need to give the index of the thalamic neuron for which we are interested here
# The neuron index is given by the keys of the dictionnary spikes
# You can extract the keys and put them in another variable with :
my_thalamus_neuron_index = list(spikes.keys())
# Now you can call the function to load HD info 
from functions import loadHDCellInfo
hd_neuron_index = loadHDCellInfo(data_directory+'Mouse12-120806_HDCells.mat', my_thalamus_neuron_index)
# You have now a new array of neuron index
# You can now separate the thalamic neurons and the head-direction thalamic neurons
# First you declare a new dictionnary
hd_spikes = {}
# Then you need to loop over the hd_neuron_index to put each neuron in the new dict
for neuron in hd_neuron_index:
	hd_spikes[neuron] = spikes[neuron]

# You can check that you have less neurons in the new dictionnary
# There are some neurons in the thalamus that are not head-direction neurons
hd_spikes.keys()

# Lastly I show you how to load the different events of the recording with one wrapper
from functions import loadEpoch
# You need to specify the type of epoch you want to load
# Possibles inputs are ['wake', 'sleep', 'sws', 'rem']
wake_ep 		= loadEpoch(data_directory, 'wake')
sleep_ep 		= loadEpoch(data_directory, 'sleep')
sws_ep 			= loadEpoch(data_directory, 'sws')
rem_ep 			= loadEpoch(data_directory, 'rem')
# The function will automaticaly search for the rigth file
# You can check your variables by typing them :
wake_ep 
sleep_ep
sws_ep 	
rem_ep 	

# you should try now to find a way to plot all the epoch to see how they follow or superpose each others