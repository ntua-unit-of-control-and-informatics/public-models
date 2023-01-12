def get_dataset(name, group):
    """
    Helper function to retrieve a TDC dataset.

    :param name: Name of the dataset
    :param path: Where to store the dataset (and search if it already exists)

    :return: The benchamark dataset and the benchmark name
    """

    # Get the Benchmark dataset
    benchmark = group.get(name)

    # all benchmark names in a benchmark group are stored in group.dataset_names
    benchmark_name = benchmark['name']

    return benchmark, benchmark_name
