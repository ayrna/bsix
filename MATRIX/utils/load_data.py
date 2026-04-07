import numpy as np
import pandas as pd

def _load_data_csv(data_dir, dataset_name, test_size, validation_size, random_state=0):
    from sklearn.model_selection import train_test_split

    print(f"\n- - - - {dataset_name} (csv) - - - -\n")

    # Load dataset
    df = pd.read_csv(f"./{data_dir}/{dataset_name}")
    df = df[df["time"] > 0]
    df = df.dropna()

    # Print dataset information
    df.info()
    print()
    print(df.describe(include="all"))
    print()

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)

    remove = ["event", "time"]
    feature_names = [x for x in df.columns.to_list() if x not in remove]
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df[["event", "time"]], test_size=test_size, random_state=random_state, stratify=df["event"])
    X_test = np.array(X_test, np.float32)
    y_test = np.array(y_test, np.float32)

    # Split dataset into train and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state, stratify=y_train["event"])
    X_train = np.array(X_train, np.float32)   
    y_train = np.array(y_train, np.float32)
    X_validation = np.array(X_validation, np.float32)
    y_validation = np.array(y_validation, np.float32)

    return X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names

def _sort_data(x, t, e):
    # Sort data by time in descending order
    sort_idx = np.argsort(t)[::-1]

    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, t, e

def _transformTrainValidationTest(X, y, dataset_name):
    from sksurv.util import Surv

    _X = X

    # y = [[event, time], [event, time], ...] / 
    # y = [[event, time_start, event_stop], [event, time_start, event_stop], ...] / 
    # y = [[event1, time1, event2, time2, ...], [event1, time1, event2, time2, ...], ...]
    _y = y
        
    if ".time_varying" in dataset_name:
        _yE = np.array([event[0] for event in _y], np.float32)
        _yTstart = np.array([time[1] for time in _y], np.float32)
        _yTstop = np.array([time[2] for time in _y], np.float32)
        _yT = np.array([_yTstart, _yTstop])
        
        # Sort data by time in descending order (during training)
        dtype = [("event", "?"), ("time_start", "f8"), ("time_stop", "f8"), ("time", "f8")]
        _y = np.array([(bool(item[0]), float(item[1]), float(item[2]), float(item[3])) for item in _y], dtype=dtype)

    elif ".multitask" in dataset_name:
        _yE = np.array(_y[:, 0::2], np.float32)
        _yT = np.array(_y[:, 1::2], np.float32)

        _ySurv = []
        for i in range(int(_y.shape[1] / 2)):
            # Sort data by time in descending order (during training)
            _ySurv.append(Surv.from_arrays(event=_yE[:, i], time=_yT[:, i]))
        _y = np.array(_ySurv).T

    else:
        _yE = np.array([event[0] for event in _y], np.float32)
        _yT = np.array([time[1] for time in _y], np.float32)

        # Sort data by time in descending order
        _X, _yT,_yE = _sort_data(_X, _yT,_yE)
        _y = Surv.from_arrays(event=_yE, time=_yT)
    
    survival = {
        "x" : _X,
        "t" : _yT,
        "e" : _yE,
    }
    
    return _X, _y, survival

def get_data(data_dir="MATRIX/datasets", dataset_name="diabetes.csv", test_size=0.2, validation_size=0.2, scaler_name="standard", scaler=None, to_multitask=False, random_state=0):
    # ".csv" is the last in the _if_ due to "analysis.csv", "multitask.csv", etc.
    if ".csv" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _load_data_csv(data_dir, dataset_name, test_size, validation_size, random_state)
    else:
        print("ERROR : Wrong format of dataset.")
        return -1

    # Scale data
    if scaler is None:
        if scaler_name == "log":
            from sklearn.preprocessing import FunctionTransformer

            def logScaler(X, shift=1.01):
                X_log = np.round(np.log(X + shift), 6)
                return X_log
            
            scaler = FunctionTransformer(func=logScaler)

        elif scaler_name == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler((0, 1))

        elif scaler_name == "standard":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

        elif scaler_name == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()

        scaler = scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)
    else:
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)
        X_test = scaler.transform(X_test)

    # Transform data for train, validation and test sets
    X_train, y_train, _ = _transformTrainValidationTest(X_train, y_train, dataset_name)
    X_validation, y_validation, _ = _transformTrainValidationTest(X_validation, y_validation, dataset_name)
    X_test, y_test, _ = _transformTrainValidationTest(X_test, y_test, dataset_name)

    # Adapt "y" for multitasking when there is only one progression
    if to_multitask:
        y_train = y_train[:, np.newaxis]
        y_validation = y_validation[:, np.newaxis]
        y_test = y_test[:, np.newaxis]
        y_test_external = y_test_external[:, np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names, scaler