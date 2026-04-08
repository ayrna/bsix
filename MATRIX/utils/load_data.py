import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def _prepare_data(df, test_size, validation_size, random_state):

    """
    Prepare the dataset for training, validation and testing.
    """
    
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

def _toDataframe(data):

    """
    Convert the HDF5 format to a DataFrame.
    """
    
    df = pd.DataFrame(data[0])
    df["event"] = data[1]
    df["time"] = data[2]

    return df

def _load_data_hdf(data_dir, dataset_name, test_size, validation_size, random_state):

    """
    Load dataset from a HDF5 file.
    """
    
    import h5py
    
    print(f"\n- - - - {dataset_name} (hdf5) - - - -\n")

    # Load dataset
    f = h5py.File(f"{data_dir}/{dataset_name}", "r")
    data = [f["x"][()], f["e"][()], f["t"][()]]
    f.close()

    df = _toDataframe(data)
    
    return _prepare_data(df, test_size, validation_size, random_state)

def _load_data_arff(data_dir, dataset_name, test_size, validation_size, random_state):

    """
    Load dataset from a ARFF file.
    """

    from scipy.io import arff

    print(f"\n- - - - {dataset_name} (arff) - - - -\n")

    # Load dataset
    file_path = f"./{data_dir}/{dataset_name}"
    data, meta = arff.loadarff(file_path)
    
    df = pd.DataFrame(data)

    # Decode byte strings to UTF-8 strings for object columns
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    return _prepare_data(df, test_size, validation_size, random_state)

def _load_data_csv(data_dir, dataset_name, test_size, validation_size, random_state):

    """
    Load dataset from a CSV file.
    """

    print(f"\n- - - - {dataset_name} (csv) - - - -\n")

    # Load dataset
    df = pd.read_csv(f"./{data_dir}/{dataset_name}")
    
    return _prepare_data(df, test_size, validation_size, random_state)

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

    """
    Load and preprocess the dataset.
    """

    if ".h5" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _load_data_hdf(data_dir, dataset_name, test_size, validation_size, random_state)
    elif ".arff" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names = _load_data_arff(data_dir, dataset_name, test_size, validation_size, random_state)
    elif ".csv" in dataset_name:
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
    X_train, y_train, _ = _transformTrainValidationTest(X_train, y_train)
    X_validation, y_validation, _ = _transformTrainValidationTest(X_validation, y_validation)
    X_test, y_test, _ = _transformTrainValidationTest(X_test, y_test)

    # Adapt "y" for multitasking when there is only one progression
    if to_multitask:
        y_train = y_train[:, np.newaxis]
        y_validation = y_validation[:, np.newaxis]
        y_test = y_test[:, np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names, scaler