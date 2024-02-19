import argparse
import time

from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset

from tdc.benchmark_group import admet_group
from src.helpers import get_dataset, cross_train_sklearn
from sklearn.svm import SVC
from jaqpotpy.doa.doa import Leverage

from src.helpers import create_featurizer
from src.helpers import create_evaluator

# Examples:
#   python x_svc.py -d AMES -f topo -s ACC AUC -c 40 -k rbf -g 0.05
#   python x_svc.py -d CYP3A4_Veith -f topo -s ACC AUPRC -c 40 -k rbf -g 0.05
#   python x_svc.py -d hERG -f maccs -s ACC AUC -c 200 -k rbf -g 0.05
#
# Handling herg needs more work as that uses a voting classifier

# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--data", required=True, help="Training data")
argParser.add_argument("-f", "--featurizer",
                       required=True,
                       choices=["mordred", "maccs", "topo"],
                       help="Molecular feature generator to use")
argParser.add_argument("-s", "--scoring-functions",
                       nargs="+",
                       choices=["MAE", "ACC", "AUC", "AUPRC"],
                       help="Scoring functions to use in the Evaluator")
argParser.add_argument("-c", "--c",
                       type=float,
                       required=True,
                       help="C value")
argParser.add_argument("-k", "--kernel",
                       choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
                       default="rbf",
                       help="kernel")
argParser.add_argument("-g", "--gamma",
                       default="scale",
                       help="kernel coefficient for rbf, poly and sigmoid. 'scale', 'auto' or a float value")
argParser.add_argument("-r", "--random-state",
                       type=int,
                       help="random state value")
argParser.add_argument("--doa",
                       help="DOA aglorithm (leverage or None)")

args = argParser.parse_args()

# the gamma value can either be the string scale or auto or a float
# see here for more info on SVC https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
if args.gamma == 'scale' or args.gamma == 'auto':
    gamma = args.gamma
else:
    gamma = float(args.gamma)

if args.doa is None:
    doa = None
elif args.doa.lower() == "leverage":
    doa = Leverage()
else:
    print("invalid value for doa. only leverage is supported. {} was specified".format(args.doa))
    exit(1)

# Get the data using the TDC client
group = admet_group(path='data/')
benchmark, name = get_dataset(args.data, group)

train_val = benchmark['train_val']
test = benchmark['test']

t0 = time.time()

# Declare the model's algorithm
svm = SVC(C=args.c, kernel=args.kernel, gamma=gamma, random_state=args.random_state)

# create the Featurizer and the Evaluator's metrics
featurizer = create_featurizer(args.featurizer)

val = create_evaluator(args.scoring_functions)

# Create a dummy Jaqpot model class
dummy_train = SmilesDataset(smiles=train_val['Drug'], y=train_val['Y'], featurizer=featurizer, task='classification')
model = MolecularSKLearn(dummy_train, doa=doa, model=svm, eval=val)

# Cross Validate and check robustness
evaluation = cross_train_sklearn(group, model, name, test, task="classification")
t1 = time.time()
print('\n\nEvaluation of the model:', evaluation)
print('Execution took {} seconds'.format(round(t1 - t0)))