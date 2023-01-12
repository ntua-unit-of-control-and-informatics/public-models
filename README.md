# public-models
This repository contains the code that is relevant to the training of the Jaqpot public models. These are 22 ML/AI models, trained on the data of the Therapeutics Data Commons ADME-Tox (ADMET) Benchmark group and are suitable for small molecule drug discovery. 

# To run

Run in the terminal the command:

python3 src/filename.py -r <run_as>

The run_as must be one of the following:
 - single (to train the model a single time)
 - cross (for cross validation of the model)
 - deploy (to cross validate the model and upload it on Jaqpot)
 
 and cannot be ommitted.
