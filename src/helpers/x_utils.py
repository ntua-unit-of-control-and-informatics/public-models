import argparse, time

from jaqpotpy.descriptors.molecular import MordredDescriptors, TopologicalFingerprint, MACCSKeysFingerprint, RDKitDescriptors
from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.doa.doa import Leverage

from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, average_precision_score
from scipy.stats import spearmanr

from tdc.benchmark_group import admet_group
from src.helpers import get_dataset, cross_train_sklearn


def create_common_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--data", required=True, help="Training data")
    argParser.add_argument("-f", "--featurizers",
                           nargs="+",
                           choices=["mordred", "maccs", "topo", "rdkit"],
                           help="Molecular feature generators to use")
    argParser.add_argument("-s", "--scoring-functions",
                           nargs="+",
                           choices=["MAE", "ACC", "AUC", "AUPRC", "SPM"],
                           help="Scoring functions to use in the Evaluator")
    argParser.add_argument("--doa",
                           help="DOA algorithm (leverage or None)")
    return argParser


def create_featurizer(name: str):
    if name == "mordred":
        return MordredDescriptors()
    elif name =='maccs':
        return MACCSKeysFingerprint()
    elif name == 'topo':
        return TopologicalFingerprint()
    elif name == 'rdkit':
        return RDKitDescriptors()
    else:
        raise ValueError("Invalid featurizer name: " + name)


def create_featurizers(names):
    return [create_featurizer(n) for n in names]


def create_evaluator(scoring_function_names):
    val = Evaluator()
    for name in scoring_function_names:
        if name == "MAE":
            val.register_scoring_function(name, mean_absolute_error)
        elif name == "ACC":
            val.register_scoring_function(name, accuracy_score)
        elif name == "AUC":
            val.register_scoring_function(name, roc_auc_score)
        elif name == "AUPRC":
            val.register_scoring_function(name, average_precision_score)
        elif name == "SPM":
            val.register_scoring_function(name, spearmanr)
        else:
            raise ValueError("Invalid scoring function name: " + name)
    return val


def create_doa(s_doa):
    """
    Domain of applicability will be Leverage() or None
    :param s_doa:
    :return:
    """
    if s_doa is None:
        return None
    elif s_doa.lower() == "leverage":
        return Leverage()
    else:
        print("invalid value for doa. only leverage is supported. {} was specified".format(args.doa))
        exit(1)


class Runner:

    def __init__(self, dataset_name: str, models: dict, doa, evaluator, featurizers, task):
        self.group = admet_group(path='data/')
        self.benchmark, self.name = get_dataset(dataset_name, self.group)

        self.train_val = self.benchmark['train_val']
        self.test = self.benchmark['test']

        self.models = models
        self.doa = doa
        self.evaluator = evaluator
        self.featurizers = featurizers
        self.task = task

    def run_cross_validation(self):

        print("Evaluating {} models".format(len(self.models)))

        t0 = time.time()
        results = []
        for featurizer in self.featurizers:
            dummy_train = SmilesDataset(smiles=self.train_val['Drug'], y=self.train_val['Y'], featurizer=featurizer, task=self.task)
            for key in self.models:
                model = self.models[key]
                skl_model = MolecularSKLearn(dummy_train, doa=self.doa, model=model, eval=self.evaluator)

                # Cross Validate and check robustness
                evaluation = cross_train_sklearn(self.group, skl_model, self.name, self.test, task=self.task)
                results.append((key + ", " + str(featurizer), evaluation))
        t1 = time.time()

        print('\n\n')
        for result in results:
            print('Evaluation of the model:', result[0], result[1])
        print('Execution took {} seconds'.format(round(t1 - t0)))
        return results
