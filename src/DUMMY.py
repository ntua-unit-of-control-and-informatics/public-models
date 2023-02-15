from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models import MolecularTorchGeometric
from jaqpotpy.datasets import TorchGraphDataset
from jaqpotpy.descriptors.molecular import AttentiveFPFeaturizer, PagtnMolGraphFeaturizer
from jaqpotpy.models.torch_models import AttentiveFP
from jaqpotpy import Jaqpot
from jaqpotpy.doa.doa import SmilesLeverage


from tdc.benchmark_group import admet_group
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
import numpy as np
from src.helpers import get_dataset, cross_train_torch
import torch
import argparse
import json


# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-r", "--run-as", help="""'single' to train the model a single time or \n
                                              'cross' for cross validation of the model or \n
                                              'deploy' to cross validate the model and upload it on Jaqpot""")
args = argParser.parse_args()


# Get the data using the TDC client
group = admet_group(path='data/')
benchmark, name = get_dataset('CYP2D6_Substrate_CarbonMangels', group)

train_val = benchmark['train_val']
test = benchmark['test']


# Declare the model's algorithm
nn = AttentiveFP(in_channels=39, hidden_channels=50, out_channels=2, edge_dim=10, num_layers=2,
                    num_timesteps=3).jittable()
optimizer = torch.optim.Adam(nn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# Declare the Featurizer and the Evaluator's metrics
featurizer = AttentiveFPFeaturizer()

val = Evaluator()
val.register_scoring_function('Accuracy', accuracy_score)
val.register_scoring_function('AUPRC', average_precision_score)
val.register_scoring_function('AUROC', roc_auc_score)


# Train model once in order to find the best algorithm and optimize it
if args.run_as == 'single':

    # Train - Validation split
    train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=42)

    # Create the Jaqpot Datasets
    jaq_train = TorchGraphDataset(smiles=train['Drug'], y=train['Y'], task='classification', featurizer=featurizer)
    jaq_train.create()

    jaq_val = TorchGraphDataset(smiles=valid['Drug'], y=valid['Y'], task='classification', featurizer=featurizer)
    jaq_val.create()

    # Update the Evaluator's dataset
    val.dataset = jaq_val

    # Train the model
    model = MolecularTorchGeometric(dataset=jaq_train
                            , model_nn=nn, eval=val, doa=None
                            , train_batch=262, test_batch=200
                            , epochs=200, optimizer=optimizer, criterion=criterion, device="cpu", test_metric=(average_precision_score, 'maximize'))
    model = model.fit()

elif args.run_as in ['cross', 'deploy']:

    # Create a dummy Jaqpot model class
    dummy_train = TorchGraphDataset(smiles=train_val['Drug'], y=train_val['Y'], task='classification', featurizer=featurizer)
    model = MolecularTorchGeometric(dataset=dummy_train
                            , model_nn=nn, eval=val, doa=None
                            , train_batch=262, test_batch=200
                            , epochs=200, optimizer=optimizer, criterion=criterion, device="cpu", test_metric=(average_precision_score, 'maximize'))

    # Cross Validate and check robustness
    evaluation = cross_train_torch(group, model, name, test, 'classification')
    print('\n\nEvaluation of the model:', evaluation)

    # Upload on Jaqpot
    if args.run_as == 'deploy':

        # Merge train and validation datasets
        train = TorchGraphDataset(smiles=train_val['Drug'], y=train_val['Y'], task='classification', featurizer=featurizer)
        train.create()

        test = TorchGraphDataset(smiles=test['Drug'], y=test['Y'], task='classification', featurizer=featurizer)
        test.create()

        # Update Evaluator's dataset
        val.dataset = test

        # Train the final model
        model = MolecularTorchGeometric(dataset=dummy_train
                                        , model_nn=nn, eval=val, doa=None
                                        , train_batch=262, test_batch=200
                                        , epochs=200, optimizer=optimizer, criterion=criterion, device="cpu", test_metric=(average_precision_score, 'maximize'))
        final_model = model.fit()

        molecular_model = final_model.create_molecular_model()

        # Jaqpot Login
        jaqpot = Jaqpot()
        jaqpot.request_key_safe()

        # Deploy model
        molecular_model.deploy_on_jaqpot(jaqpot=jaqpot,
                                     description="ADME model predicting CYP3A4 - an important enzyme in the body, mainly found in the liver and in the intestine - which oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.",
                                     model_title="CYP3A4 Substrate CarbonMangels Model")

        # Opening Submission JSON file
        with open('data/submission_results.json', 'r') as openfile:
            # Reading from json file
            submission = json.load(openfile)

        submission[name] = evaluation[name]
        with open("data/submission_results.json", "w") as outfile:
            json.dump(submission, outfile)

else:
    raise ValueError(f'Argument {args.run_as} is not acceptable. Users must provide either "single" or "cross" or "deploy"')