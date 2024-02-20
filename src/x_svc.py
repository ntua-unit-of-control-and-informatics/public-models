from tdc.benchmark_group import admet_group

from sklearn.svm import SVC

from src.helpers import create_featurizers
from src.helpers import create_evaluator
from src.helpers import create_doa
from src.helpers import create_common_args
from src.helpers import Runner

# Examples:
#   python x_svc.py -d AMES -f topo -s ACC AUC -c 40 -k rbf -g 0.05
#   python x_svc.py -d CYP3A4_Veith -f topo -s ACC AUPRC -c 40 -k rbf -g 0.05
#   python x_svc.py -d hERG -f maccs -s ACC AUC -c 200 -k rbf -g 0.05
#
# Handling herg needs more work as that uses a voting classifier

# Argument to control the execution of the code
argParser = create_common_args()
argParser.add_argument("-c", "--c",
                       nargs="+",
                       type=float,
                       required=True,
                       help="C value(s)")
argParser.add_argument("-k", "--kernels",
                       nargs="+",
                       choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
                       default=["rbf"],
                       help="kernel(s)")
argParser.add_argument("-g", "--gamma",
                       default="scale",
                       help="kernel coefficient for rbf, poly and sigmoid ('scale', 'auto' or a float value")
argParser.add_argument("-r", "--random-state",
                       type=int,
                       help="random state value")

args = argParser.parse_args()

# the gamma value can either be the string scale or auto or a float
# see here for more info on SVC https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
if args.gamma == 'scale' or args.gamma == 'auto':
    gamma = args.gamma
else:
    gamma = float(args.gamma)

doa = create_doa(args.doa)

# Declare the different variants of the model's algorithm
models = {}
for c in args.c:
    for k in args.kernels:
        model = SVC(C=c, kernel=k, gamma=gamma, random_state=args.random_state)
        key = "C={}, kernel={}".format(str(c), k)
        models[key] = model
        print("added model", key)

# create the Featurizer and the Evaluator's metrics
featurizers = create_featurizers(args.featurizers)
val = create_evaluator(args.scoring_functions)

runner = Runner(args.data, models, doa, val, featurizers, "classification")
results = runner.run_cross_validation()
