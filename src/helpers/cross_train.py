from jaqpotpy.datasets import SmilesDataset, TorchGraphDataset


def cross_train_sklearn(group, model, name, test_df, task='regression'):
    """
    'Helper function to cross validate a SKLearn model.'

    :param group: The TDC Benchmark Group Class object
    :param model: The Jaqpot model object
    :param name: Individual benchmark dataset name
    :param test_df: Pandas dataframe of the test dataset

    :return: A dictionary with the cross validated evaluation of the model
    """

    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        predictions = {}

        # Train - Validation split
        train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

        # Create the Jaqpot Datasets
        jaq_train = SmilesDataset(smiles=train['Drug'], y=train['Y'], featurizer=model.dataset.featurizer, task=task)
        jaq_train.create()

        jaq_val = SmilesDataset(smiles=valid['Drug'], y=valid['Y'], featurizer=model.dataset.featurizer, task=task)
        jaq_val.create()

        # Update the datasets
        model.evaluator.dataset = jaq_val
        model.dataset = jaq_train

        # Train the model
        trained_model = model.fit()

        # Take predictions on the Test set
        trained_model(test_df['Drug'].tolist())

        # Keep the predictions
        predictions[name] = trained_model.prediction
        predictions_list.append(predictions)

    # Return the cross validation score
    return group.evaluate_many(predictions_list)


def cross_train_torch(group, nn, name, test_df, task='regression'):
    """
    'Helper function to cross validate a Torch model.'

    :param group: The TDC Benchmark Group Class object
    :param model: The Jaqpot model object
    :param name: Individual benchmark dataset name
    :param test_df: Pandas dataframe of the test dataset

    :return: A dictionary with the cross validated evaluation of the model
    """

    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        predictions = {}

        # Train - Validation split
        train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

        # Create the Jaqpot Datasets
        jaq_train = TorchGraphDataset(smiles=train['Drug'], y=train['Y'], featurizer=nn.dataset.featurizer, task=task)
        jaq_train.create()

        jaq_val = TorchGraphDataset(smiles=valid['Drug'], y=valid['Y'], featurizer=nn.dataset.featurizer, task=task)
        jaq_val.create()

        # Update the datasets
        nn.evaluator.dataset = jaq_val
        nn.dataset = jaq_train

        # Train the model
        trained_model = nn.fit()

        # Create molecular model and take predictions on the Test set
        mol_model = trained_model.create_molecular_model()
        mol_model(test_df['Drug'].tolist())

        # Keep the predictions
        predictions[name] = mol_model.prediction
        predictions_list.append(predictions)

    # Return the cross validation score
    return group.evaluate_many(predictions_list)