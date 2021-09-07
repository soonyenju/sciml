from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split


def split(Xs, ys, return_index = False):
    if return_index:
        sss = ShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        sss.get_n_splits(Xs, ys)
        train_index, test_index = next(sss.split(Xs, ys)) 
        return (train_index, test_index)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, 
            test_size = 0.33, 
            random_state = 42
        )
        return (X_train, X_test, y_train, y_test)