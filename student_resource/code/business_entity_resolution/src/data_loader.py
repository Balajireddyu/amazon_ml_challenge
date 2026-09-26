import pandas as pd


def load_training_data(data_dir):
    source1 = pd.read_csv(
        f"{data_dir}/train/train_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        f"{data_dir}/train/train_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        f"{data_dir}/train/train_source3.tsv",
        sep="\t"
    )

    ground_truth = pd.read_csv(
        f"{data_dir}/train/train_ground_truth.tsv",
        sep="\t"
    )

    return source1, source2, source3, ground_truth