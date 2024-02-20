from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint
from jaqpotpy import Jaqpot

from tdc.benchmark_group import admet_group
from sklearn.metrics import mean_absolute_error
from src.helpers import get_dataset, cross_train_sklearn
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.doa.doa import Leverage
import argparse
import json


# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-r", "--run-as", help="""'single' to train the model a single time or \n
                                              'cross' for cross validation of the model or \n
                                              'deploy' to cross validate the model and upload it on Jaqpot or \n
                                              'save' to save the model to a local file""")
args = argParser.parse_args()

NAME = 'ppbr_az'
# Get the data using the TDC client
group = admet_group(path = 'data/')
benchmark, name = get_dataset('PPBR_AZ', group)

train_val = benchmark['train_val']
test = benchmark['test']

# Declare the model's algorithm
rf = RandomForestRegressor(n_estimators=100, min_samples_split=4, max_depth=7, random_state=8)

# Declare the Featurizer and the Evaluator's metrics
featurizer = MACCSKeysFingerprint()

val = Evaluator()
val.register_scoring_function('MAE', mean_absolute_error)


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
    model = MolecularSKLearn(jaq_train, doa=Leverage(), model=rf, eval=val)
    _ = model.fit()

elif args.run_as in ['cross', 'deploy', 'save']:

    # Create a dummy Jaqpot model class
    dummy_train = SmilesDataset(smiles=train_val['Drug'], y=train_val['Y'], featurizer=featurizer)
    model = MolecularSKLearn(dummy_train, doa=Leverage(), model=rf, eval=val)

    # Cross Validate and check robustness
    evaluation = cross_train_sklearn(group, model, name, test)
    print('\n\nEvaluation of the model:', evaluation)

    # Upload on Jaqpot
    if args.run_as == 'deploy' or args.run_as == 'save':

        # Merge train and validation datasets
        train = SmilesDataset(smiles = train_val['Drug'], y = train_val['Y'], featurizer = featurizer)
        train.create()

        test = SmilesDataset(smiles = test['Drug'], y = test['Y'], featurizer = featurizer)
        test.create()

        # Update Evaluator's dataset
        val.dataset = test

        # Train the final model
        model = MolecularSKLearn(train, doa=Leverage(), model=rf, eval=val)
        final_model = model.fit()

        if args.run_as == 'deploy':

            # Jaqpot Login
            jaqpot = Jaqpot()
            jaqpot.request_key_safe()

            # Deploy model
            final_model.deploy_on_jaqpot(jaqpot=jaqpot,
                                         description="ADME model predicting the human plasma protein binding rate (PPBR), which is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery.",
                                         model_title="PPBR Model")

            # Opening Submission JSON file
            with open('data/submission_results.json', 'r') as openfile:
                # Reading from json file
                submission = json.load(openfile)

            submission[name] = evaluation[name]
            with open("data/submission_results.json", "w") as outfile:
                json.dump(submission, outfile)

        elif args.run_as == 'save':

            print("Saving model to {}.jmodel".format(NAME))
            final_model.model_name = NAME
            final_model.model_title = NAME  # title is used as the base of the filename
            final_model.save()

else:
    raise ValueError(f'Argument {args.run_as} is not acceptable. Users must provide either "single" or "cross" or "deploy"')