import argparse
import time

from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset

from tdc.benchmark_group import admet_group
from src.helpers import get_dataset, cross_train_sklearn
from sklearn.naive_bayes import BernoulliNB
from jaqpotpy.doa.doa import Leverage

from src.helpers import create_featurizer
from src.helpers import create_evaluator

# Examples:
#   python x_bernoullinb.py -d CYP2C9_Substrate_CarbonMangels -f maccs -s ACC AUPRC -a 0.1
#   python x_bernoullinb.py -d CYP2C9_Veith -f topo -s ACC AUPRC
#   python x_bernoullinb.py -d CYP2D6_Veith -f topo -s ACC AUPRC
#
# Handling half life and clearance hepatocyte needs more work as those use a voting classifier
#
# See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

# Argument to control the execution of the code
argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--data", required=True, help="Training data")
argParser.add_argument("-f", "--featurizer",
                       required=True,
                       choices=["mordred", "maccs", "topo"],
                       help="Molecular feature generator to use")
argParser.add_argument("-s", "--scoring-functions",
                       nargs="+",
                       choices=["MAE", "ACC", "AUC", "AUPRC", "SPM"],
                       help="Scoring functions to use in the Evaluator")
argParser.add_argument("-a", "--alpha",
                       type=float,
                       default=1.0,
                       help="alpha value")
argParser.add_argument("--doa",
                       help="DOA aglorithm (leverage or None)")

args = argParser.parse_args()

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
bnb = BernoulliNB(alpha=args.alpha)

# create the Featurizer and the Evaluator's metrics
featurizer = create_featurizer(args.featurizer)

val = create_evaluator(args.scoring_functions)

# Create a dummy Jaqpot model class
dummy_train = SmilesDataset(smiles=train_val['Drug'], y=train_val['Y'], featurizer=featurizer, task='regression')
model = MolecularSKLearn(dummy_train, doa=doa, model=bnb, eval=val)

# Cross Validate and check robustness
evaluation = cross_train_sklearn(group, model, name, test, task="regression")
t1 = time.time()
print('\n\nEvaluation of the model:', evaluation)
print('Execution took {} seconds'.format(round(t1 - t0)))