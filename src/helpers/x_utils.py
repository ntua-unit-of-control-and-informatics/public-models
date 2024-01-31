from jaqpotpy.descriptors.molecular import MordredDescriptors, TopologicalFingerprint, MACCSKeysFingerprint
from jaqpotpy.models.evaluator import Evaluator

from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score


def create_featurizer(name: str):
    if name == "mordred":
        return MordredDescriptors()
    elif name =='maccs':
        return MACCSKeysFingerprint()
    elif name == 'topo':
        return TopologicalFingerprint()
    else:
        raise ValueError("Invalid featurizer name: " + name)


def create_evaluator(scoring_function_names):
    val = Evaluator()
    for name in scoring_function_names:
        if name == "MAE":
            val.register_scoring_function(name, mean_absolute_error)
        elif name == "ACC":
            val.register_scoring_function(name, accuracy_score)
        elif name == "AUC":
            val.register_scoring_function(name, roc_auc_score)
        else:
            raise ValueError("Invalid scoring function name: " + name)
    return val