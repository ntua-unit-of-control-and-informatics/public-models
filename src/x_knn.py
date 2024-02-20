from tdc.benchmark_group import admet_group

from sklearn.neighbors import KNeighborsClassifier

from src.helpers import create_featurizers
from src.helpers import create_evaluator
from src.helpers import create_doa
from src.helpers import create_common_args
from src.helpers import Runner

# Example:
#   python x_knn.py -d BBB_Martins -f mordred -s ACC AUC --n-neighbours 3

# Argument to control the execution of the code
argParser = create_common_args()
argParser.add_argument("-n", "--n-neighbours", nargs="+",
                       type=int, default=[3], help="Num neighbours")

args = argParser.parse_args()

doa = create_doa(args.doa)

# Declare the different variants of the model's algorithm
models = {}
for n in args.n_neighbours:
    model = KNeighborsClassifier(n_neighbors=n)
    key = "n_neighbours={}".format(str(n))
    models[key] = model
    print("added model", key)

# create the Featurizer and the Evaluator's metrics
featurizers = create_featurizers(args.featurizers)
val = create_evaluator(args.scoring_functions)

runner = Runner(args.data, models, doa, val, featurizers, "classification")
results = runner.run_cross_validation()
