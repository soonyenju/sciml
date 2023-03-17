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