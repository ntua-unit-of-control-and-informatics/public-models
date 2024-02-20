from tdc.benchmark_group import admet_group
from sklearn.ensemble import RandomForestRegressor

from src.helpers import create_featurizers
from src.helpers import create_evaluator
from src.helpers import create_doa
from src.helpers import create_common_args
from src.helpers import Runner

# Example usage:
#   python x_rf.py -d Caco2_Wang -f mordred --max-depth 9 -s MAE --task regression

# Argument to control the execution of the code
argParser = create_common_args()
argParser.add_argument("--n-estimators", nargs="+", type=int, default=[200], help="Num RF estimators")
argParser.add_argument("--min-samples-split", nargs="+", type=int, default=[5], help="Min RF samples split")
argParser.add_argument("--max-depth", nargs="+", type=int, default=[9], help="Max RF depth")
argParser.add_argument("--random-state", type=int, default=8, help="RF random state")
argParser.add_argument("-t", "--task",
                       required=True,
                       choices=["regression", "classification"],
                       help="regression or classification model")

args = argParser.parse_args()

doa = create_doa(args.doa)

# Declare the different variants of the model's algorithm
models = {}
for e in args.n_estimators:
    for s in args.min_samples_split:
        for d in args.max_depth:
            model = RandomForestRegressor(n_estimators=e,
                                          min_samples_split=s,
                                          max_depth=d,
                                          random_state=args.random_state)
            key = "n_estimators={}, min_samples_split={}, max_depth={}".format(str(e), str(s), str(d))
            models[key] = model
            print("added model", key)

# create the Featurizer and the Evaluator's metrics
featurizers = create_featurizers(args.featurizers)
val = create_evaluator(args.scoring_functions)

runner = Runner(args.data, models, doa, val, featurizers, args.task)
results = runner.run_cross_validation()

