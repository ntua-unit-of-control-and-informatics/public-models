from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import TopologicalFingerprint, MACCSKeysFingerprint
from jaqpotpy import Jaqpot

from tdc.benchmark_group import admet_group
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from src.helpers import get_dataset, cross_train_sklearn
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np
import argparse
import json


# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-r", "--run-as", help="""'single' to train the model a single time or \n
                                              'cross' for cross validation of the model or \n
                                              'deploy' to cross validate the model and upload it on Jaqpot""")
args = argParser.parse_args()


# Get the data using the TDC client
group = admet_group(path = 'data/')
benchmark, name = get_dataset('Clearance_Hepatocyte_AZ', group)

train_val = benchmark['train_val']
test = benchmark['test']


# Declare the model's algorithm
svm = SVR(C=60, kernel='poly', gamma=0.01)
sgd = SGDRegressor(loss='epsilon_insensitive', epsilon=2.5, random_state=10)
voter = VotingRegressor(estimators=[('svm', svm), ('sgd', sgd)], weights=np.array([0.75, 0.25]))


# Declare the Featurizer and the Evaluator's metrics
featurizer = MACCSKeysFingerprint()

val = Evaluator()
val.register_scoring_function('MAE', mean_absolute_error)
val.register_scoring_function('SPM', spearmanr)


# Train model once in order to find the best algorithm and optimize it
if args.run_as == 'single':

    # Train - Validation split
    train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = 42)

    # Create the Jaqpot Datasets
    jaq_train = SmilesDataset(smiles = train['Drug'], y = train['Y'], featurizer = featurizer)
    jaq_train.create()

    jaq_val = SmilesDataset(smiles = valid['Drug'], y = valid['Y'], featurizer = featurizer)
    jaq_val.create()

    # Update the Evaluator's dataset
    val.dataset = jaq_val

    # Train the model
    model = MolecularSKLearn(jaq_train, doa=None, model=voter, eval=val)
    _ = model.fit()

elif args.run_as in ['cross', 'deploy']:

    # Create a dummy Jaqpot model class
    dummy_train = SmilesDataset(smiles=train_val['Drug'], y=train_val['Y'], featurizer=featurizer)
    model = MolecularSKLearn(dummy_train, doa=None, model=voter, eval=val)

    # Cross Validate and check robustness
    evaluation = cross_train_sklearn(group, model, name, test)
    print('\n\nEvaluation of the model:', evaluation)

    # Upload on Jaqpot
    if args.run_as == 'deploy':

        # Merge train and validation datasets
        train = SmilesDataset(smiles = train_val['Drug'], y = train_val['Y'], featurizer = featurizer)
        train.create()

        test = SmilesDataset(smiles = test['Drug'], y = test['Y'], featurizer = featurizer)
        test.create()

        # Update Evaluator's dataset
        val.dataset = test

        # Train the final model
        model = MolecularSKLearn(train, doa=None, model=voter, eval=val)
        final_model = model.fit()

        # Jaqpot Login
        jaqpot = Jaqpot()
        jaqpot.request_key_safe()

        # Deploy model
        final_model.deploy_on_jaqpot(jaqpot=jaqpot,
                                     description="ADME model predicting the volume of plasma cleared of a drug over a specified time period, which measures the rate at which the active drug is removed from the body.",
                                     model_title="Clearance Hepatocyte Model")

        # Opening Submission JSON file
        with open('data/submission_results.json', 'r') as openfile:
            # Reading from json file
            submission = json.load(openfile)

        submission[name] = evaluation[name]
        with open("data/submission_results.json", "w") as outfile:
            json.dump(submission, outfile)

else:
    raise ValueError(f'Argument {args.run_as} is not acceptable. Users must provide either "single" or "cross" or "deploy"')