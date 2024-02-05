from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset

from tdc.benchmark_group import admet_group
from src.helpers import get_dataset, cross_train_sklearn
from sklearn.ensemble import RandomForestRegressor
from jaqpotpy.doa.doa import Leverage
import argparse

from src.helpers import create_featurizer
from src.helpers import create_evaluator

# Example usage:
#   python x_rf.py -d Caco2_Wang -f mordred --max-depth 9 --task regression

# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--data", required=True, help="Training data")
argParser.add_argument("-f", "--featurizer",
                       required=True,
                       choices=["mordred", "maccs", "topo"],
                       help="Molecular feature generator to use")
argParser.add_argument("--n-estimators", type=int, default=200, help="Num RF estimators")
argParser.add_argument("--min-samples-split", type=int, default=5, help="Min RF samples split")
argParser.add_argument("--max-depth", type=int, default=9, help="Max RF depth")
argParser.add_argument("--random-state", type=int, default=8, help="RF random state")
argParser.add_argument("-t", "--task",
                       required=True,
                       choices=["regression", "classification"],
                       help="regression or classification model")

args = argParser.parse_args()

# Get the data using the TDC client
group = admet_group(path='data/')
benchmark, name = get_dataset(args.data, group)

train_val = benchmark['train_val']
test = benchmark['test']

# Declare the model's algorithm
rf = RandomForestRegressor(n_estimators=args.n_estimators,
                           min_samples_split=args.min_samples_split,
                           max_depth=args.max_depth,
                           random_state=args.random_state)

# create the Featurizer and the Evaluator's metrics
featurizer = create_featurizer(args.featurizer)

val = create_evaluator(["MAE"])

# Create a dummy Jaqpot model class
dummy_train = SmilesDataset(smiles=train_val['Drug'], y=train_val['Y'], featurizer=featurizer, task=args.task)
model = MolecularSKLearn(dummy_train, doa=Leverage(), model=rf, eval=val)

# Cross Validate and check robustness
evaluation = cross_train_sklearn(group, model, name, test, task=args.task)
print('\n\nEvaluation of the model:', evaluation)


