import numpy as np
import pandas as pd
import re

from scipy.stats import mode
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

def _prepare_data(df, test_size, validation_size, seed):

    """
    Prepare the dataset for training, validation and testing.
    """

    from sklearn.model_selection import StratifiedGroupKFold

    # Remove rows with 0.0 or less in time columns
    if all(name in df.columns for name in ["time_start", "time_stop"]):
        event_labels = ["event"]
        time_labels = ["time_start", "time_stop"]

        labels = event_labels + time_labels

        # For time-varying data, only filter on time_stop (the event/censoring time)
        df = df[df["time_stop"] > 0]
    elif all(name in df.columns for name in ["event1", "time1"]):
        event_labels = df.filter(regex=r'^event\d+$').columns.tolist()
        time_labels = df.filter(regex=r'^time\d+$').columns.tolist()

        labels = event_labels + time_labels
        labels.sort(key=lambda x: int(re.search(r'(\d+)$', x).group()))

        df = df[(df[time_labels] >= 0).all(axis=1)]
    else:
        event_labels = ["event"]
        time_labels = ["time"]

        labels = event_labels + time_labels

        df = df[df["time"] > 0]
    
    df = df.dropna()
    df = df.reset_index(drop=True)

    index = df.index

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
    # Standard data
    if len(event_labels) == 1 and len(time_labels) == 1:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(df[feature_names], df[labels], index, test_size=test_size, random_state=seed, stratify=df[event_labels])
        
        X_train, X_validation, y_train, y_validation, train_idx, val_idx = train_test_split(X_train, y_train, train_idx, test_size=validation_size, random_state=seed, stratify=y_train[event_labels])

        X_train = np.array(X_train, np.float32)   
        y_train = np.array(y_train, np.float32)
        X_validation = np.array(X_validation, np.float32)
        y_validation = np.array(y_validation, np.float32)
        X_test = np.array(X_test, np.float32)
        y_test = np.array(y_test, np.float32)
    # Time varying and multitask data
    elif len(event_labels) >= 1 and len(time_labels) > 1:
        n_groups = df["identifier"].nunique()
        n_splits_outer = min(int(1 / test_size), max(2, n_groups // 2))
        n_splits_inner = min(int(1 / validation_size), max(2, n_groups // 4))
        
        group_split_outer = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
        if len(event_labels) == 1: # Time varying
            train_val_idx, test_idx = next(group_split_outer.split(df[feature_names], df[event_labels], groups=df["identifier"]))
        elif len(event_labels) > 1: # Multitask (multi-progression)
            stratify_train_test = df[event_labels].astype(str).agg('_'.join, axis=1)
            train_val_idx, test_idx = next(group_split_outer.split(df[feature_names], stratify_train_test, groups=df["identifier"]))

        X_train_val = df.iloc[train_val_idx]
        y_train_val = df.iloc[train_val_idx][labels]
        X_test = df.iloc[test_idx][feature_names]
        y_test = df.iloc[test_idx][labels]

        group_split_inner = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
        if len(event_labels) == 1: # Time varying
            train_idx, val_idx = next(group_split_inner.split(X_train_val, y_train_val[event_labels], groups=X_train_val["identifier"]))
        elif len(event_labels) > 1: # Multitask (multi-progression)
            stratify_train_val = y_train_val[event_labels].astype(str).agg('_'.join, axis=1)
            train_idx, val_idx = next(group_split_inner.split(X_train_val, stratify_train_val, groups=X_train_val["identifier"]))

        if "identifier" in feature_names:
            feature_names.remove("identifier")
            
        X_train = np.array(X_train_val.iloc[train_idx][feature_names].values, np.float32)
        y_train = np.array(y_train_val.iloc[train_idx][labels].values, np.float32)
        X_validation = np.array(X_train_val.iloc[val_idx][feature_names].values, np.float32)
        y_validation = np.array(y_train_val.iloc[val_idx][labels].values, np.float32)
        X_test = np.array(X_test[feature_names].values, np.float32)
        y_test = np.array(y_test[labels].values, np.float32)

    return X_train, y_train, X_validation, y_validation, X_test, y_test, train_idx, val_idx, test_idx, feature_names

def _toDataframe(data):

    """
    Convert the HDF5 format to a DataFrame.
    """
    
    df = pd.DataFrame(data[0])

    # Time varying data
    if all(name in df.columns for name in ["time_start", "time_stop"]): 
        df["event"] = data[1]
        df["time_start"] = data[2][:, 0]
        df["time_stop"] = data[2][:, 1]
    # Multitask (multi-progression) data
    elif all(name in df.columns for name in ["event1", "time1"]): 
        progression_colon = len(df.filter(regex=r'^event\d+$').columns.tolist())
        for i in range(progression_colon):
            df[f"event{i+1}"] = data[1][:, i]
            df[f"time{i+1}"] = data[2][:, i]
    # Standard data
    else:
        df["event"] = data[1]
        df["time"] = data[2]

    return df

def load_data_hdf(data_dir, dataset_name):

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

def load_data_arff(data_dir, dataset_name):

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

def load_data_csv(data_dir, dataset_name):

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

    survival_X = X.copy()
    
    # Standard: [event, time]
    if y.shape[1] == 2:
        _yE = y[:, 0].astype(np.float32)
        _yT = y[:, 1].astype(np.float32)

        survival_X, _yT, _yE = _sort_data(X, _yT, _yE)
        survival_y = Surv.from_arrays(event=_yE, time=_yT)
    # Time varying: [event, time_start, time_stop]
    elif y.shape[1] == 3: 
        _yE = y[:, 0].astype(np.float32)
        _yTstart = y[:, 1].astype(np.float32)
        _yTstop = y[:, 2].astype(np.float32)
        _yT = np.array([_yTstart, _yTstop])
        
        dtype = [('event', '?'), ('time_start', 'f8'), ('time_stop', 'f8')]
        survival_y = np.empty(len(y), dtype=dtype)
        survival_y['event'] = y[:, 0].astype(bool)
        survival_y['time_start'] = y[:, 1].astype(float)
        survival_y['time_stop'] = y[:, 2].astype(float)
    # Multitask (multi-progression): [event1, time1, event2, time2, ...]
    else:
        _yE = y[:, 0::2].astype(np.float32) # (0, 2, 4...)
        _yT = y[:, 1::2].astype(np.float32) # (1, 3, 5...)

        number_progressions = y.shape[1] // 2
        surv_list = [Surv.from_arrays(event=_yE[:, i], time=_yT[:, i]) for i in range(number_progressions)]
        survival_y = np.array(surv_list).T
    
    survival = {
        "x" : survival_X,
        "t" : _yT,
        "e" : _yE,
    }
    
    return survival_X, survival_y, survival

def _filter_low_variance(X, threshold=0.80):

    """
    Filter out columns with low variance.
    """

    _, counts = mode(X, axis=0, keepdims=True)
    mask = (counts[0] / X.shape[0]) <= threshold

    return mask

def _filter_high_correlation(X, threshold=0.80):

    """
    Filter out columns with high correlation.
    """

    corr_matrix = np.nan_to_num(np.abs(np.corrcoef(X.T)))
        
    upper = np.triu(corr_matrix, k=1)
    drop_indices = np.unique(np.where(upper > threshold)[1])
    
    mask = np.ones(X.shape[1], dtype=bool)
    mask[drop_indices] = False
    
    return mask

def _filter_high_vif(X, threshold=5.0):

    """
    Filter out columns with high Variance Inflation Factor (VIF).
    """
    
    vif_values = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        vif_values[i] = variance_inflation_factor(X, i)
            
    mask = vif_values <= threshold
    
    return mask

def get_data(df=None, data_dir="MATRIX/datasets", dataset_name="colon.csv", test_size=0.2, validation_size=0.2, scaler_name="standard", scaler=None, to_multitask=False, seed=0):

    """
    Load and preprocess the dataset.
    """

    if df is not None:
        X_train, y_train, X_validation, y_validation, X_test, y_test, train_idx, val_idx, test_idx, feature_names = _prepare_data(df, test_size, validation_size, seed)
    elif ".h5" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, train_idx, val_idx, test_idx, feature_names = _prepare_data(load_data_hdf(data_dir, dataset_name), test_size, validation_size, seed)
    elif ".arff" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, train_idx, val_idx, test_idx, feature_names = _prepare_data(load_data_arff(data_dir, dataset_name), test_size, validation_size, seed)
    elif ".csv" in dataset_name:
        X_train, y_train, X_validation, y_validation, X_test, y_test, train_idx, val_idx, test_idx, feature_names = _prepare_data(load_data_csv(data_dir, dataset_name), test_size, validation_size, seed)
    else:
        print("ERROR : Wrong format of dataset.")
        return -1

    mask = _filter_low_variance(X_train)
    X_train = X_train[:, mask]
    X_validation = X_validation[:, mask]
    X_test = X_test[:, mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    if X_train.shape[1] > 1:
        mask = _filter_high_correlation(X_train)
        X_train = X_train[:, mask]
        X_validation = X_validation[:, mask]
        X_test = X_test[:, mask]
        feature_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    if X_train.shape[1] > 1:
        mask = _filter_high_vif(X_train)
        X_train = X_train[:, mask]
        X_validation = X_validation[:, mask]
        X_test = X_test[:, mask]
        feature_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        
    # Convert to DataFrame for scaling
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_validation_df = pd.DataFrame(X_validation, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Scale data
    if scaler is None:
        if scaler_name == "log":
            from sklearn.preprocessing import FunctionTransformer

            def logScaler(X, shift=(1 + 1e-6)):
                X_log = np.round(np.log(X + shift), 6)
                return X_log
            
            scaler = FunctionTransformer(func=logScaler).set_output(transform="pandas")

        elif scaler_name == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler((0, 1)).set_output(transform="pandas")

        elif scaler_name == "standard":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().set_output(transform="pandas")

        elif scaler_name == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler().set_output(transform="pandas")

        scaler = scaler.fit(X_train_df)
        
        X_train_df = scaler.transform(X_train_df)
        X_validation_df = scaler.transform(X_validation_df)
        X_test_df = scaler.transform(X_test_df)
    else:
        X_train_df = scaler.transform(X_train_df)
        X_validation_df = scaler.transform(X_validation_df)
        X_test_df = scaler.transform(X_test_df)

    # Convert back to numpy arrays
    X_train = np.array(X_train_df.values, np.float32)
    X_validation = np.array(X_validation_df.values, np.float32)
    X_test = np.array(X_test_df.values, np.float32)
    
    # Transform data for train, validation and test sets
    X_train, y_train, _ = _transformTrainValidationTest(X_train, y_train)
    X_validation, y_validation, _ = _transformTrainValidationTest(X_validation, y_validation)
    X_test, y_test, _ = _transformTrainValidationTest(X_test, y_test)

    # Adapt "y" for multitasking when there is only one progression
    if to_multitask and y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
        y_validation = y_validation[:, np.newaxis]
        y_test = y_test[:, np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, train_idx, val_idx, test_idx, feature_names, scaler