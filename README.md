# StarterPack
- to learn python and neuroseries quickly. 
- This package is intended for the Peyrache Lab internal usage. 
- Scripts should be read and tested in your favorite python environment in the following order:

1. [main1_basics](https://github.com/PeyracheLab/StarterPack/blob/master/python/main1_basics.py) - *Nice and gentle walktrough of python, numpy and matplotlib.*
2. [main2_neuroseries](https://github.com/PeyracheLab/StarterPack/blob/master/python/main2_neuroseries.py) - *Introduction to neuroseries for handling spike times, time series and epochs.*
3. [main3_tuningcurves](https://github.com/PeyracheLab/StarterPack/blob/master/python/main3_tuningcurves.py) - *How to make an angular tuning curve?*
4. [main4_raw_data](https://github.com/PeyracheLab/StarterPack/blob/master/python/main4_raw_data.py) - *How to load data coming from the preprocessing pipeline (i.e. .res, .clu files)?*
5. [main5_matlab_data](https://github.com/PeyracheLab/StarterPack/blob/master/python/main5_matlab_data.py) - *Too bad, Adrien asked you to analyse his old data saved in matlab...*
6. [main6_autocorr](https://github.com/PeyracheLab/StarterPack/blob/master/python/main6_autocorr.py) - *How to make an auto-correlogram ?*
7. [main7_replay](https://github.com/PeyracheLab/StarterPack/blob/master/python/main7_replay.py) - *How to do bayesian decoding?*



# Requirements
- python 3
- numpy 
- scipy
- pandas
- matplotlib
- numba
- pycircstats
- pytables
- nose
- neuroseries (It's best to use version located inside the StarterPack)

## Example data

The example session KA28-190405 that should be saved in /StarterPack/data_raw/ can be found [here](https://www.dropbox.com/sh/cuz6x9g0ru3bqvo/AACubJBC4gseHLmBOmY7h8mVa?dl=1)
