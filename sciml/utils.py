import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

# randomly select sites
def random_select(ds, count, num, random_state = 0):
    np.random.seed(random_state)
    idxs = np.random.choice(np.delete(np.arange(len(ds)), count), num, replace = False)
    return np.sort(idxs)

def split(Xs, ys, return_index = False, test_size = 0.33, random_state = 42):
    if return_index:
        sss = ShuffleSplit(n_splits=1, test_size = test_size, random_state = random_state)
        sss.get_n_splits(Xs, ys)
        train_index, test_index = next(sss.split(Xs, ys)) 
        return (train_index, test_index)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, 
            test_size = test_size, 
            random_state = random_state
        )
        return (X_train, X_test, y_train, y_test)
    
def split_cut(Xs, ys, test_ratio = 0.33):
    """
    Split the timeseries into before and after halves
    """
    assert ys.ndim == 2, 'ys must be 2D!'
    assert len(Xs) == len(ys), 'Xs and ys should be equally long!'
    assert type(Xs) == type(ys), 'Xs and ys should be the same data type!'
    if not type(Xs) in [pd.core.frame.DataFrame, np.ndarray]: raise Exception('Only accept numpy ndarray or pandas dataframe')
    anchor = int(np.floor(len(ys) * (1 - test_ratio)))

    if type(Xs) == pd.core.frame.DataFrame:
        X_train = Xs.iloc[0: anchor, :]
        X_test = Xs.iloc[anchor::, :]
        y_train = ys.iloc[0: anchor, :]
        y_test = ys.iloc[anchor::, :]
    else:
        X_train = Xs[0: anchor, :]
        X_test = Xs[anchor::, :]
        y_train = ys[0: anchor, :]
        y_test = ys[anchor::, :]

    assert len(X_train) + len(X_test) == len(Xs), 'The sum of train and test lengths must equal to Xs/ys!'

    return (X_train, X_test, y_train, y_test)