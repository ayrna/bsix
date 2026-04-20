import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def _prepare_data(df, test_size, validation_size, seed):

    """
    Prepare the dataset for training, validation and testing.
    """

    import numpy.lib.recfunctions as rfn
    from sklearn.model_selection import StratifiedGroupKFold
    
    if all(name in df.columns for name in ["time_start", "time_stop"]):
        time_labels = ["time_start", "time_stop"]
        # For time-varying data, only filter on time_stop (the event/censoring time)
        df = df[df["time_stop"] > 0]
    else:
        time_labels = ["time"]
        df = df[df["time"] > 0]
        
    labels = ["event"] + time_labels
    df = df.dropna()
    
    # Print dataset information
    df.info()
    print()
    print(df.describe(include="all"))
    print()

    # One-hot encoding for categorical variables (excluding labels)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove label columns from categorical encoding
    categorical_cols = [col for col in categorical_cols if col not in labels]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    feature_names = [x for x in df.columns.to_list() if x not in labels]
    
    # Split dataset into train and test sets
    if len(time_labels) == 1:
        X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df[labels], test_size=test_size, random_state=seed, stratify=df["event"])

        X_test = np.array(X_test, np.float32)
        y_test = np.array(y_test, np.float32)

        # Split dataset into train and validation sets
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, random_state=seed, stratify=y_train["event"])

        X_train = np.array(X_train, np.float32)   
        y_train = np.array(y_train, np.float32)
        X_validation = np.array(X_validation, np.float32)
        y_validation = np.array(y_validation, np.float32)
    else:
        n_groups = df["identifier"].nunique()
        n_splits_outer = min(int(1 / test_size), max(2, n_groups // 2))
        n_splits_inner = min(int(1 / validation_size), max(2, n_groups // 4))
        
        group_split_outer = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(group_split_outer.split(df[feature_names], df["event"], groups=df["identifier"]))

        X_train_val = df.iloc[train_val_idx]
        y_train_val = df.iloc[train_val_idx][labels]
        X_test = df.iloc[test_idx][feature_names]
        y_test = df.iloc[test_idx][labels]

        group_split_inner = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
        train_idx, val_idx = next(group_split_inner.split(X_train_val, y_train_val["event"], groups=X_train_val["identifier"]))

        if "identifier" in feature_names:
            feature_names.remove("identifier")
            
        X_train = np.array(X_train_val.iloc[train_idx][feature_names].values, np.float32)
        y_train = np.array(y_train_val.iloc[train_idx][labels].values, np.float32)
        X_validation = np.array(X_train_val.iloc[val_idx][feature_names].values, np.float32)
        y_validation = np.array(y_train_val.iloc[val_idx][labels].values, np.float32)
        X_test = np.array(X_test[feature_names].values, np.float32)
        y_test = np.array(y_test[labels].values, np.float32)

    return X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names

def _toDataframe(data):

    """
    Convert the HDF5 format to a DataFrame.
    """
    
    df = pd.DataFrame(data[0])
    df["event"] = data[1]
    df["time"] = data[2]

    return df

def _load_data_hdf(data_dir, dataset_name):

    """
    Load dataset from a HDF5 file.
    """
    
    import h5py

    # Load dataset
    f = h5py.File(f"{data_dir}/{dataset_name}", "r")
    data = [f["x"][()], f["e"][()], f["t"][()]]
    f.close()

    df = _toDataframe(data)

    print(f"\n- - - - {dataset_name} (hdf5) - - - -\n")
    
    return df

def _load_data_arff(data_dir, dataset_name):

    """
    Load dataset from a ARFF file.
    """

    from scipy.io import arff

    # Load dataset
    file_path = f"./{data_dir}/{dataset_name}"
    data, meta = arff.loadarff(file_path)
    
    df = pd.DataFrame(data)

    # Decode byte strings to UTF-8 strings for object columns
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    print(f"\n- - - - {dataset_name} (arff) - - - -\n")
    
    return df

def _load_data_csv(data_dir, dataset_name):

    """
    Load dataset from a CSV file.
    """

    # Load dataset
    df = pd.read_csv(f"./{data_dir}/{dataset_name}")

    print(f"\n- - - - {dataset_name} (csv) - - - -\n")
    
    return df

def _sort_data(x, t, e):

    """
    Sort data by time in descending order.
    """
    
    sort_idx = np.argsort(t)[::-1]

    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, t, e

def _transformTrainValidationTest(X, y):

    """
    Transform the data format for train, validation and test sets.
    """

    from sksurv.util import Surv

    _X = X

    # y = [[event, time], [event, time], ...] / 
    # y = [[event, time_start, event_stop], [event, time_start, event_stop], ...] / 
    # y = [[event1, time1, event2, time2, ...], [event1, time1, event2, time2, ...], ...]
    _y = y
    
    if y.shape[1] == 2:
        _yE = np.array([event[0] for event in _y], np.float32)
        _yT = np.array([time[1] for time in _y], np.float32)

        # Sort data by time in descending order
        _X, _yT,_yE = _sort_data(_X, _yT,_yE)
        _y = Surv.from_arrays(event=_yE, time=_yT)
    else:
        _yE = np.array([event[0] for event in _y], np.float32)#.reshape(-1)
        _yTstart = np.array([time[1] for time in _y], np.float32)#.reshape(-1)
        _yTstop = np.array([time[2] for time in _y], np.float32)#.reshape(-1)
        _yT = np.array([_yTstart, _yTstop])
        # Ordenar posteriormente por tiempo descendente (ahora no)
        dtype = [('event', '?'), ('time_start', 'f8'), ('time_stop', 'f8')]
        _y = np.array([(bool(item[0]), float(item[1]), float(item[2])) for item in _y], dtype=dtype)
    
    survival = {
        "x" : _X,
        "t" : _yT,
        "e" : _yE,
    }
    
    return _X, _y, survival

def get_data(df=None, data_dir="MATRIX/datasets", dataset_name="colon.csv", test_size=0.2, validation_size=0.2, scaler_name="standard", scaler=None, to_multitask=False, seed=0):

    """
    Load and preprocess the dataset.
    """

    if df is not None:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _prepare_data(df, test_size, validation_size, seed)
    elif ".h5" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _prepare_data(_load_data_hdf(data_dir, dataset_name), test_size, validation_size, seed)
    elif ".arff" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _prepare_data(_load_data_arff(data_dir, dataset_name), test_size, validation_size, seed)
    elif ".csv" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _prepare_data(_load_data_csv(data_dir, dataset_name), test_size, validation_size, seed)
    else:
        print("ERROR : Wrong format of dataset.")
        return -1

    # Scale data
    if scaler is None:
        if scaler_name == "log":
            from sklearn.preprocessing import FunctionTransformer

            def logScaler(X, shift=(1 + 1e-6)):
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
    X_train, y_train, _ = _transformTrainValidationTest(X_train, y_train)
    X_validation, y_validation, _ = _transformTrainValidationTest(X_validation, y_validation)
    X_test, y_test, _ = _transformTrainValidationTest(X_test, y_test)

    # Adapt "y" for multitasking when there is only one progression
    if to_multitask:
        y_train = y_train[:, np.newaxis]
        y_validation = y_validation[:, np.newaxis]
        y_test = y_test[:, np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names, scaler