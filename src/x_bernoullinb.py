from tdc.benchmark_group import admet_group

from sklearn.naive_bayes import BernoulliNB

from src.helpers import create_featurizers
from src.helpers import create_evaluator
from src.helpers import create_doa
from src.helpers import create_common_args
from src.helpers import Runner

# Examples:
#   python x_bernoullinb.py -d CYP2C9_Substrate_CarbonMangels -f maccs -s ACC AUPRC -a 0.1
#   python x_bernoullinb.py -d CYP2C9_Veith -f topo -s ACC AUPRC
#   python x_bernoullinb.py -d CYP2D6_Veith -f topo -s ACC AUPRC
#
# Handling half life and clearance hepatocyte needs more work as those use a voting classifier
#
# See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

# Argument to control the execution of the code
argParser = create_common_args()
argParser.add_argument("-a", "--alphas",
                       type=float,
                       nargs="+",
                       default=[1.0],
                       help="alpha value(s)")

args = argParser.parse_args()

doa = create_doa(args.doa)

# Declare the different variants of the model's algorithm
models = {}
for a in args.alphas:
    model = BernoulliNB(alpha=a)
    key = "alpha={}".format(str(a))
    models[key] = model
    print("added model", key)


# create the Featurizer and the Evaluator's metrics
featurizers = create_featurizers(args.featurizers)
val = create_evaluator(args.scoring_functions)

runner = Runner(args.data, models, doa, val, featurizers, "classification")
results = runner.run_cross_validation()
