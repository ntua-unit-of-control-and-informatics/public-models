from tdc.benchmark_group import admet_group
from sklearn.svm import SVR

from src.helpers import create_featurizers
from src.helpers import create_evaluator
from src.helpers import create_doa
from src.helpers import create_common_args
from src.helpers import Runner

# Examples:
#   python x_svr.py -d Clearance_Microsome_AZ -f topo -s MAE SPM -c 80 -k poly -g 0.01
#   python x_svr.py -d Lipophilicity_AstraZeneca -f topo -s MAE -c 50 -k rbf -g 0.01
#   python x_svr.py -d Solubility_AqSolDB -f maccs -s MAE -c 50 -k rbf -g 0.1
#   python x_svr.py -d VDss_Lombardo -f topo -s MAE SPM -c 60 -k poly -g 0.01
#
# Handling half life and clearance hepatocyte needs more work as those use a voting classifier

# Argument to control the execution of the code
argParser = create_common_args()
argParser.add_argument("-c", "--c",
                       type=float,
                       nargs="+",
                       help="C value(s)")
argParser.add_argument("-k", "--kernels",
                       nargs="+",
                       choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
                       default=["rbf"],
                       help="kernel(s)")
argParser.add_argument("-g", "--gamma",
                       default="scale",
                       help="kernel coefficient for rbf, poly and sigmoid. 'scale', 'auto' or a float value")

args = argParser.parse_args()

# the gamma value can either be the string scale or auto or a float
# see here for more info on SVR https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
if args.gamma == 'scale' or args.gamma == 'auto':
    gamma = args.gamma
else:
    gamma = float(args.gamma)

doa = create_doa(args.doa)

# Declare the different variants of the model's algorithm
models = {}
for c in args.c:
    for k in args.kernels:
        model = SVR(C=c, kernel=k, gamma=gamma)
        key = "C={}, kernel={}".format(str(c), k)
        models[key] = model
        print("added model", key)


# create the Featurizer and the Evaluator's metrics
featurizers = create_featurizers(args.featurizers)
val = create_evaluator(args.scoring_functions)

runner = Runner(args.data, models, doa, val, featurizers, "regression")
results = runner.run_cross_validation()
