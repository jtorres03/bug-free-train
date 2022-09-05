import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # Read in training data
    df = pd.read_csv("input/train.csv")

    # Create new column 'kfolds' equal to -1
    df['kfold'] = -1

    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold clas from model_selection module
    # we use stratified kfold to kep the same % of targets per fold
    # (because of skewed targets)
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # fill the new kfold column
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    # save to new csv with kfold column
    df.to_csv('input/train_folds.csv', index=False)
