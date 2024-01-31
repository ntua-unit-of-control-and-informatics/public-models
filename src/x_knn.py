from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset

from tdc.benchmark_group import admet_group
from src.helpers import get_dataset, cross_train_sklearn
from sklearn.neighbors import KNeighborsClassifier
from jaqpotpy.doa.doa import Leverage
import argparse

from src.helpers import create_featurizer
from src.helpers import create_evaluator

# Example:
#   python x_knn.py -d BBB_Martins -f mordred -s ACC AUC --n-neighbours 3

# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--data", required=True, help="Training data")
argParser.add_argument("-f", "--featurizer",
                       required=True,
                       choices=["mordred", "maccs", "topo"],
                       help="Molecular feature generator to use")
argParser.add_argument("-s", "--scoring-functions",
                       nargs="+",
                       choices=["MAE", "ACC", "AUC"],
                       help="Scoring functions to use in the Evaluator")
argParser.add_argument("-t", "--task",
                       required=True,
                       choices=["regression", "classification"],
                       help="regression or classification model")
argParser.add_argument("-n", "--n-neighbours", type=int, default=3, help="Num neighbours")

args = argParser.parse_args()

# Get the data using the TDC client
group = admet_group(path='data/')
benchmark, name = get_dataset(args.data, group)

train_val = benchmark['train_val']
test = benchmark['test']

# Declare the model's algorithm
knn = KNeighborsClassifier(n_neighbors=3)

# create the Featurizer and the Evaluator's metrics
featurizer = create_featurizer(args.featurizer)

val = create_evaluator(args.scoring_functions)

# Create a dummy Jaqpot model class
dummy_train = SmilesDataset(smiles=train_val['Drug'], y=train_val['Y'], featurizer=featurizer)
model = MolecularSKLearn(dummy_train, doa=Leverage(), model=knn, eval=val)

# Cross Validate and check robustness
evaluation = cross_train_sklearn(group, model, name, test, task=args.task)
print('\n\nEvaluation of the model:', evaluation)
