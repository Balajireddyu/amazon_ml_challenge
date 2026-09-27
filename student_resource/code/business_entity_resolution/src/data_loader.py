import os
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


def load_test_data(data_dir):
    s1_path = os.path.join(data_dir, "test", "test_source1.tsv")
    if not os.path.exists(s1_path):
        s1_path = os.path.join(data_dir, "test", "source1.tsv")

    source1 = pd.read_csv(s1_path, sep="\t")

    s2_path = os.path.join(data_dir, "test", "test_source2.tsv")
    if not os.path.exists(s2_path):
        s2_path = os.path.join(data_dir, "test", "source2.tsv")
    if not os.path.exists(s2_path):
        s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    source2 = pd.read_csv(s2_path, sep="\t")

    s3_path = os.path.join(data_dir, "test", "test_source3.tsv")
    if not os.path.exists(s3_path):
        s3_path = os.path.join(data_dir, "test", "source3.tsv")
    if not os.path.exists(s3_path):
        s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    source3 = pd.read_csv(s3_path, sep="\t")

    return source1, source2, source3