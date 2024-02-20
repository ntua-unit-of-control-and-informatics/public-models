
from tdc.benchmark_group import admet_group

from sklearn.linear_model import LogisticRegression

from src.helpers import create_featurizers
from src.helpers import create_evaluator
from src.helpers import create_doa
from src.helpers import create_common_args
from src.helpers import Runner

# Examples:
#   python x_lr.py -d HIA_Hou -f maccs -s ACC AUC -c 50 -p elasticnet --solver saga -l 0.5 --doa leverage
#
# See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

# Argument to control the execution of the code
argParser = create_common_args()
argParser.add_argument("-c", "--c",
                       type=float,
                       nargs="+",
                       default=[1.0],
                       help="C value(s)")
argParser.add_argument("-p", "--penalty",
                       choices=["l1", "l2", "elasticnet"],
                       help="penalty")
argParser.add_argument("--solvers",
                       nargs="+",
                       choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                       default=["lbfgs"],
                       help="algorithm(s) to use in the optimization problem")
argParser.add_argument("-l", "--l1-ratio",
                       type=float,
                       help="The Elastic-Net mixing parameter (between 0 and 1), default None")
argParser.add_argument("-r", "--random-state",
                       type=int,
                       help="random state value")

args = argParser.parse_args()

doa = create_doa(args.doa)

# Declare the different variants of the model's algorithm
models = {}
for c in args.c:
    for s in args.solvers:
        model = LogisticRegression(C=c, penalty=args.penalty, solver=s, l1_ratio=args.l1_ratio, random_state=args.random_state)
        key = "C={}, solver={}".format(str(c), s)
        models[key] = model
        print("added model", key)

# create the Featurizer and the Evaluator's metrics
featurizers = create_featurizers(args.featurizers)
val = create_evaluator(args.scoring_functions)



runner = Runner(args.data, models, doa, val, featurizers, "classification")
results = runner.run_cross_validation()

