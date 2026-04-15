import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch

from abc import ABC, abstractmethod
from lifelines import CoxPHFitter, KaplanMeierFitter, PiecewiseExponentialFitter, statistics
from scipy.stats import entropy
from sklearn.base import BaseEstimator

def _tool_setTimeTicksAxisX(ax):

    """
    Tool for setting time ticks on X-axis.
    """

    max_days = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])) # (min, max)

    if max_days > 3650: # More than 10 years
        major, minor = 1825, 365 # 5 years - 1 year
    elif max_days > 365: # Between 1 year and 10 years
        major, minor = 365, 73 # 1 year - 5 splits per year
    else: # Less than 1 year
        major, minor = 30, 6 # 1 month - 5 splits per month

    return major, minor

def _tool_setXaiTicksAxisX(ax):

    """
    Tool for setting XAI ticks on X-axis.
    """

    max_shap = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])) # (min, max)

    if max_shap > 10: # More than 10 (xai)
        major, minor = 1, 0.25 # 1 - 0.25 (xai)
    elif max_shap > 1: # More than 1 (xai)
        major, minor = 0.5, 0.1 # 0.5 - 0.1 (xai)
    else: # Less than 1 (xai)
        major, minor = 0.1, 0.05 # 0.1 - 0.05 (xai)

    return major, minor

def _tool_setRiskTicksAxisY(ax):

    """
    Tool for setting risk ticks on Y-axis.
    """

    max_risk = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) # (min, max)

    if max_risk > 5: # More than 5 (risk)
        major, minor = 1, 0.25 # 1 - 0.5 (risk)
    else: # Less than 5 (risk)
        major, minor = 0.5, 0.1 # 0.2 - 0.1 (risk)

    return major, minor

def _tool_toDataframe(data, columns=None):

    """
    Tool for converting X, y to a DataFrame.
    """

    if columns == None: # Without columns names
        dataframe = pd.DataFrame(data, columns=[str(l) for l in range(data.shape[1])])
    else: # With columns names
        dataframe = pd.DataFrame(data, columns=columns)

    return dataframe
    
class BaseSurvival(BaseEstimator, ABC):
    
    """
    Abstract Class for Survival Analysis models.
    """

    @abstractmethod
    def calculate_xai(self, X, **kwargs):

        """
        Calculate XAI values.
        """

        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X, y, **kwargs):

        """
        Fit the model.
        """

        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X, **kwargs):

        """
        Predict on X.
        """

        raise NotImplementedError
    
    @abstractmethod
    def predict_cumulative_hazard_function(self, X, **kwargs):

        """
        H(x,t) = H0(t) * exp(g(x)).
        """

        raise NotImplementedError
    
    @abstractmethod
    def predict_survival_function(self, X, **kwargs):

        """
        S(x, t) = exp(-H(x, t)).
        """

        raise NotImplementedError
    
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    @staticmethod
    def dinamic_discretise(y, dataset, seed):

        """
        Discretise data by piecewise exponential and show in kaplan meier.
        """

        rng = np.random.default_rng(seed=seed)

        # Piecewise Exponential (constant risk at intervals) #
        # Risk groups = num splits + 1
        num_splits = [3, 4, 5]

        # Define the search range
        # n_iter = 100 is the ideal number of iterations (tested)
        n_iter = 5 # Iterations in search (random search)
        min_time = np.quantile(y["time"], 0.05) #y["time"].min()
        max_time = np.quantile(y["time"], 0.95) #y["time"].max()
        possible_splits = np.arange(min_time, max_time, int((max_time - min_time) / 15)).tolist() # 15 splits
        
        # Search (random search)
        best_splits = None
        
        results = []
        for n_splits in num_splits:
            for _ in range(n_iter):
                # Randomized split = (n1, n2, n3, ...)
                split = sorted(rng.choice(possible_splits, n_splits, replace=False))
                # Instantiate the Piecewise Exponential model with breakpoints
                pf = PiecewiseExponentialFitter(breakpoints=list(split))
                # Fit
                pf.fit(y["time"], y["event"])
                
                # Calculate the size of each split (only events)
                splits = [0] + list(split) + [np.inf]
                lengths = pd.cut(y["time"][y["event"] == 1], bins=splits).value_counts()

                # Save the results
                results.append({
                    "n_splits": n_splits,
                    "split": split,
                    "aic": pf.AIC_,
                    "entropy": (1.0 - (entropy(lengths) / np.log(n_splits + 1))),
                })

        # Convert to dataframe
        df_results = pd.DataFrame(results)

        # Apply Min-Max to normalize
        # AIC
        min_aic = df_results["aic"].min()
        max_aic = df_results["aic"].max()
        df_results["aic_norm"] = (df_results["aic"] - min_aic) / (max_aic - min_aic)

        # Entropy
        min_entropy = df_results["entropy"].min()
        max_entropy = df_results["entropy"].max()
        df_results["entropy_norm"] = (df_results["entropy"] - min_entropy) / (max_entropy - min_entropy)

        # Calculate the score (weighted)
        alpha = 0.5
        df_results["score"] = (alpha * df_results["aic_norm"]) + ((1 - alpha) * df_results["entropy_norm"])

        # Save the best split
        best_splits = df_results["split"].loc[df_results["score"].idxmin()]

        # Kaplan Meier #
        # Instantiate the Kapaln Meier estimator
        kmf = KaplanMeierFitter()
        # Fit
        kmf.fit(durations=y["time"], event_observed=y["event"])

        # Plot #
        # Configure style
        plt.figure(figsize=(10, 6))

        # Personalise curve
        ax = kmf.plot(
            color="#C1502E",
            label=f"KM estimate"
        )

        # Splits
        for split in best_splits:
            plt.axvline(x=split, color="#2EC192", linestyle="-.", alpha=0.5)

        # Title and axis labels
        ax.set_title(f"Discretised Kaplan-Meier\n{dataset} - seed {seed}", fontsize=12)
        ax.set_xlabel("Time (days)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)

        # Axis ticks
        majorX, minorX = _tool_setTimeTicksAxisX(ax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(majorX))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minorX))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        plt.xticks(rotation=45, ha="right")

        # Axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Grid
        plt.grid(True, which="major", linestyle="-", alpha=0.7)
        plt.grid(True, which="minor", linestyle="--", alpha=0.7, linewidth=0.5)

        # Legend
        plt.legend(frameon=True, facecolor="white", edgecolor="0.8")

        # Save figure
        plt.tight_layout()
        plt.savefig(f"Plot_DiscretisedKM-{dataset}_s{seed}.png", bbox_inches="tight", dpi=300)
        plt.close()

        splits = [0] + best_splits + [np.inf]

        return splits

    @staticmethod
    def feature_selection(X, y):

        """
        Calculate the best features based on p-value.
        """
        
        # Cox model (lifelines)
        model = CoxPHFitter(penalizer=0.01)

        labels_covariables = ["event", "time"]
        # Transform data to dataframe
        dataframe = pd.concat([_tool_toDataframe(X), _tool_toDataframe(y, labels_covariables)], axis=1)
        # List of significance covariables (p-value)
        significance_covariables = [c for c in dataframe.columns.to_list() if c not in labels_covariables]
        
        # Fit with all the covariables
        model.fit(dataframe[significance_covariables + labels_covariables], duration_col=labels_covariables[1], event_col=labels_covariables[0], show_progress=False)
        # Compute p-values (sinificance)
        significance = model._compute_p_values()
        
        # Obtain the significance covariables (p-value < 0.05)
        while max(significance) >= 0.05 and len(significance_covariables) > 0:
            # Delete the covariable with the maximum p-value
            covariables_to_delete = significance_covariables[np.where(significance == max(significance))[0][0]]
            significance_covariables.remove(covariables_to_delete)
            
            # Fit with the significance covariables
            model.fit(dataframe[significance_covariables + labels_covariables], duration_col=labels_covariables[1], event_col=labels_covariables[0], show_progress=False)
            # Compute p-values (sinificance)
            significance = model._compute_p_values()
        
        # Obtain the index of the significance covariables
        significance_covariables = [int(i) for i in significance_covariables]

        return significance_covariables
    
    @staticmethod
    def generate_simulated_survival_data(number_rows=1000, number_columns=10, censored=0.75, relation=None, seed=0):

        """
        Generate simulated survival data based.
        """

        # Fix the seed
        np.random.seed(seed)
            
        # Generate covariates (normal distribution [nature])
        X = np.random.normal(0, 1, size=(number_rows, number_columns))
        names_columns = [f"feature_{i}" for i in range(number_columns)]
        
        # Generate coeffs (uniform distribution [same probability])
        coeffs = np.random.uniform(-1, 1, size=number_columns)
        
        # Calculate log_risk (lineal or non-lineal)

        # Lineal: H(x) = beta * x
        log_risk = np.dot(X, coeffs)

        if relation == "cuadratic":
            # Cuadratic: H(x) = beta * x^2
            log_risk = np.dot(X**2, coeffs)
            
        elif relation == "sin":
            # Sin: H(x) = beta * sin(x * pi)
            log_risk = np.dot(np.sin(X * np.pi), coeffs)
        
        # Calculate hazard risk
        risk = np.exp(log_risk)
        
        # Calculate survival times (S(t) = e^(−λt) where λ = baseline × risk)
        baseline = 0.15
        S = np.random.uniform(0, 1, size=number_rows)
        time_survival = -np.log(S) / (baseline * risk)
        
        # Calculate censored individuals
        number_censored = int(round(number_rows * censored))
        idx_censored = np.random.choice(number_rows, size=number_censored, replace=False)
        
        # Event
        event = np.ones(number_rows, dtype=int)

        # Time
        time_observed = time_survival.copy()
        time_censored = np.random.uniform(0, time_survival[idx_censored])
        
        time_observed[idx_censored] = time_censored
        event[idx_censored] = 0
        
        # Dataframe
        dataframe = pd.DataFrame(X, columns=names_columns)
        dataframe["event"] = event
        dataframe["time"] = np.round(time_observed, 2)
        
        return dataframe
    
    @staticmethod
    def logrank_test(y, groups, weights=None):

        """
        Calculate the log-rank test for n groups.
        """

        result = statistics.multivariate_logrank_test(y["time"], groups, y["event"], weights)
        result.print_summary()

        return result
    
    @staticmethod
    def to_time_dependent(dataframe, identifier, splits, time="time", event="event"):
    
        """
        Transform a DataFrame with a per-subject measurement into a time-dependent format.
        """

        # Sort dataframe by identifier
        dataframe_transformed = dataframe.sort_values(by=[identifier]).copy()
        # Rename columns
        dataframe_transformed = dataframe_transformed.rename(columns={identifier: "identifier", event: "event", time: "time"})

        # Aply discretisation
        dataframe_transformed["time_frame"] = pd.cut(dataframe_transformed["time"], bins=splits, labels=False)

        # Repeat each row N times according to the time_frame column (time discretised)
        dataframe_transformed = dataframe_transformed.loc[dataframe_transformed.index.repeat(dataframe_transformed["time_frame"] + 1)]
        # Accumulate form 0 to time_discretised value
        dataframe_transformed["time_frame"] = dataframe_transformed.groupby("identifier").cumcount()

        # Reset index (repeat step)
        dataframe_transformed = dataframe_transformed.reset_index(drop=True)

        # Last index of the row for each patient
        last_row_index = dataframe_transformed.groupby("identifier").tail(1).index
        # Indicate whether (or not) the event occurred in the last row for each patient
        dataframe_transformed.loc[~dataframe_transformed.index.isin(last_row_index), "event"] = 0.0

        # Calculate the split associated with the time_frame value
        dataframe_transformed["days_risk"] = dataframe_transformed["time_frame"].map(dict(enumerate(splits)))
        # Indicate the real value of time  in the last row for each patient
        dataframe_transformed.loc[last_row_index, "days_risk"] = dataframe_transformed.loc[last_row_index, "time"]
        
        return dataframe_transformed
    
    @staticmethod
    def to_time_varying(dataframe, identifier, time="time", event="event"):
    
        """
        Transform a DataFrame with a multiple-subject measurements into a start-stop format.
        """
        
        # Sort dataframe by identifier and date
        dataframe_transformed = dataframe.sort_values(by=[identifier, time]).copy()
        # Rename columns
        dataframe_transformed = dataframe_transformed.rename(columns={identifier: "identifier", event: "event", time: "time_stop"})

        # Move the new time_start column (time) down by inserting 0.0 as the first value
        dataframe_transformed["time_start"] = dataframe_transformed.groupby("identifier")["time_stop"].shift(1).fillna(0)
        dataframe_transformed = dataframe_transformed.astype({"time_start": float, "time_stop": float})

        # Move the event column down by inserting 0.0 as the first value (do not remove the row with the event)
        shift_event = dataframe_transformed.groupby("identifier")["event"].shift(1).fillna(0).astype(int)
        # Pass the event down the chain
        forward_fill_event = shift_event.groupby(dataframe_transformed["identifier"]).cummax()
        # Remove events with a value of 1.0
        dataframe_transformed = dataframe_transformed[forward_fill_event == 0]
        
        # Reorder dataframe
        cols = [col for col in dataframe_transformed.columns if col not in ["identifier", "time_start", "time_stop", "event"]]
        dataframe_transformed = dataframe_transformed[["identifier"] + cols + ["event", "time_start", "time_stop"]]

        # Rest index
        dataframe_transformed = dataframe_transformed.reset_index(drop=True)
        
        return dataframe_transformed
    
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    @staticmethod
    def plot_coefficients(coefficients, estimator_name, dataset, seed=None, progression=None):

        """
        Plot XAI coefficients for the data.
        """

        # Extract data
        values = np.array(list(coefficients.values()), np.float32)
        names = np.array(list(coefficients.keys()), str)
        
        # Sort features by importance
        importance = np.abs(values)
        sort_idx = np.argsort(importance)

        names = names[sort_idx]
        values = values[sort_idx]

        # Configure style
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("coolwarm")

        # Normalise the color
        max_abs = np.nanmax(np.abs(values)) + 1e-6
        normalise = plt.Normalize(vmin=-max_abs, vmax=max_abs)

        # Obtain color map
        color = cmap(normalise(values))

        ax.barh(names, values, color=color, edgecolor="#000000", alpha=0.8, zorder=3) # z-ordering for layers

        # Draw vertical line (xaxis = 0)
        ax.axvline(x=0, color="#000000", linewidth=0.75, zorder=2) # z-ordering for layers

        # Title and axis labels
        title_parts = [f"{estimator_name} - {dataset}"]
        if seed is not None:
            title_parts.append(f"seed {seed}")
        if progression is not None:
            title_parts.append(f"progression {progression}")
        
        plt.title(f"XAI\n{' - '.join(title_parts)}", fontsize=12)
        plt.xlabel("Coefficients values", fontsize=10)
        plt.ylabel("Features", fontsize=10)
        
        # Axis ticks
        ax = plt.gca()
        majorX, minorX = _tool_setXaiTicksAxisX(ax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(majorX))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minorX))

        plt.xticks(rotation=45, ha="right")

        # Grid
        plt.grid(True, which="major", linestyle="-", alpha=0.7, zorder=0) # z-ordering for layers
        plt.grid(True, which="minor", linestyle="--", alpha=0.7, linewidth=0.5, zorder=0) # z-ordering for layers

        # Build filename dynamically
        filename_parts = [f"Plot_XAI_coefficients-{estimator_name}_{dataset}"]
        if seed is not None:
            filename_parts.append(f"s{seed}")
        if progression is not None:
            filename_parts.append(f"p{progression}")
        filename = f"{'_'.join(filename_parts)}.png"
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def plot_shap(shap_explainer, estimator_name, dataset, seed=None, progression=None):

        """
        Plot SHAP values for the data.
        """

        # Extract data
        values = shap_explainer.values
        data = shap_explainer.data
        names = np.array(shap_explainer.feature_names, str)

        # Sort features by importance
        importance = np.abs(values).mean(axis=0)
        sort_idx = np.argsort(importance)

        # Configure style
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("coolwarm")

        # Plot points
        for y_pos, idx in enumerate(sort_idx, start=1):
            x = values[:, idx]       
            x_original = data[:, idx] 
            
            # Normalise the color
            min_val = np.nanmin(x_original)
            max_val = np.nanmax(x_original) + 1e-6
            
            # Jitter
            y = y_pos + np.random.normal(0, 0.075, size=len(x))
            
            ax.scatter(x, y, s=10, c=x_original, cmap=cmap, vmin=min_val, vmax=max_val, alpha=0.8, edgecolors="none", zorder=3) # z-ordering for layers

        # Add the color bar (legend)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        color_bar = fig.colorbar(sm, ax=ax)
        color_bar.set_label("Feature value", labelpad=-15, fontsize=10)
        color_bar.set_ticks([0, 1])
        color_bar.set_ticklabels(["Low", "High"])

        # Draw vertical line (xaxis = 0)
        ax.axvline(x=0, color="#000000", linewidth=0.75, zorder=2) # z-ordering for layers

        # Title and axis labels
        title_parts = [f"{estimator_name} - {dataset}"]
        if seed is not None:
            title_parts.append(f"seed {seed}")
        if progression is not None:
            title_parts.append(f"progression {progression}")
        
        plt.title(f"XAI\n{' - '.join(title_parts)}", fontsize=12)
        plt.xlabel("Shap values", fontsize=10)
        plt.ylabel("Features", fontsize=10)

        # Axis ticks
        ax = plt.gca()
        majorX, minorX = _tool_setXaiTicksAxisX(ax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(majorX))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minorX))

        plt.xticks(rotation=45, ha="right")
        plt.yticks(ticks=np.arange(1, len(sort_idx) + 1), labels=names[sort_idx])

        # Grid
        plt.grid(True, which="major", linestyle="-", alpha=0.7, zorder=0) # z-ordering for layers
        plt.grid(True, which="minor", linestyle="--", alpha=0.7, linewidth=0.5, zorder=0) # z-ordering for layers

        # Build filename dynamically
        filename_parts = [f"Plot_XAI_values-{estimator_name}_{dataset}"]
        if seed is not None:
            filename_parts.append(f"s{seed}")
        if progression is not None:
            filename_parts.append(f"p{progression}")
        filename = f"{'_'.join(filename_parts)}.png"
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _plot_survival_hazard_functions(X, estimator_name, dataset, seed, function_type, progression=None):

        """
        Plot survival and cumulative hazard functions for the data.
        """

        # Configure style
        plt.figure(figsize=(10, 6))
        
        # Plot curve
        for _, step_function in enumerate(X):
            times = step_function.x
            probabilities = step_function(times)
            
            plt.step(times, probabilities, where="post", alpha=0.6)
        
        # Title and axis labels
        title_parts = [f"{estimator_name} - {dataset}"]
        if seed is not None:
            title_parts.append(f"seed {seed}")
        if progression is not None:
            title_parts.append(f"progression {progression}")
        
        plt.title(f"{function_type}\n{' - '.join(title_parts)}", fontsize=12)
        plt.xlabel("Time (days)", fontsize=10)
        plt.ylabel(f"{function_type} probability", fontsize=10)
        
        # Axis ticks
        ax = plt.gca()
        majorX, minorX = _tool_setTimeTicksAxisX(ax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(majorX))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minorX))

        if function_type == "Survival":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        if function_type == "CumulativeRisk":
            majorY, minorY = _tool_setRiskTicksAxisY(ax)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(majorY))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(minorY))

        plt.xticks(rotation=45, ha="right")

        # Axis limits
        ax.set_xlim(left=0)

        if function_type == "Survival":
            ax.set_ylim(bottom=0, top=1.05)

        if function_type == "CumulativeRisk":
            ax.set_ylim(bottom=0)

        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Grid
        plt.grid(True, which="major", linestyle="-", alpha=0.7)
        plt.grid(True, which="minor", linestyle="--", alpha=0.7, linewidth=0.5)
        
        # Build filename dynamically
        filename_parts = [f"Plot_{function_type}-{estimator_name}_{dataset}"]
        if seed is not None:
            filename_parts.append(f"s{seed}")
        if progression is not None:
            filename_parts.append(f"p{progression}")
        filename = f"{'_'.join(filename_parts)}.png"
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _sort(self, X, y, time="time"):
        
        """
        Sort data by descending time.
        """
                
        sort_idx = np.argsort(y[time])[::-1]

        X = X[sort_idx]
        y = y[sort_idx]

        return X, y
    
    def _sort_multitask(self, risk, t, e):

        """
        Sort data by descending time (multitask).
        """
        
        _, idx = torch.sort(t, descending=True)

        risk = risk[idx]
        t = t[idx]
        e = e[idx]
        
        return risk, t, e