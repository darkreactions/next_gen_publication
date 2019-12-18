from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def configure_input_df(train_set):
    all_cols = train_set.columns.tolist()
    # don't worry about _calc_ columns for now, but it's in the code
    #  so they get included once the data is available
    feature_cols = [c for c in all_cols if (
        "_rxn_" in c) or ("_feat_" in c) or ("_calc_" in c)]
    non_numerical_cols = (train_set.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]

    # Convert crystal scores 1-3 to 0 and 4 to 1. i.e. Binarizing scores
    conditions = [
        (train_set['_out_crystalscore'] == 1),
        (train_set['_out_crystalscore'] == 2),
        (train_set['_out_crystalscore'] == 3),
        (train_set['_out_crystalscore'] == 4),
    ]
    binarized_labels = [0, 0, 0, 1]

    # Add a column called binarized_crystalscore which is the column to predict
    train_set['binarized_crystalscore'] = np.select(
        conditions, binarized_labels)
    col_order = list(train_set.columns.values)
    col_order.insert(3, col_order.pop(
        col_order.index('binarized_crystalscore')))
    train_set = train_set[col_order]
    return train_set, feature_cols


def generate_80_20_splits(train_set):
    train_set, feature_cols = configure_input_df(train_set)
    train, test = train_test_split(train_set, test_size=0.2, random_state=5,
                                   stratify=train_set[['dataset']])
    return train, test


if __name__ == '__main__':
    # path to dataset
    dataset_path = './0046.perovskitedata.csv'
    df = pd.read_csv(dataset_path)

    # 2 pandas dataframes representing training and testing set
    train_set, test_set = generate_80_20_splits(df)
