#!/usr/bin/env python

'''
	File name: main4_raw_data.py
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

This scripts will show you how to use the wrappers function to load raw data
In a typical 

The data should be found in StarterPack/data_matlab/

This script will show you how to load the various data you need
The function are already written in the file functions.py that should be in the same directory

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