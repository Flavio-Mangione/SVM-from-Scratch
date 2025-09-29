from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Literal,List
import numpy as np
from SVMKit import SVM
from sklearn.model_selection import KFold
import pandas as pd
from IPython.display import display
from typing import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

def build_svm(kernel_type: str, C: float, parameter: dict, solver: str, decision_function_shape=None) -> SVM:
    if kernel_type == 'gaussian':
        return SVM(kernel='gaussian', C=C, gamma=parameter['gamma'],
                   solver=solver, decision_function_shape=decision_function_shape)
    elif kernel_type == 'polynomial':
        return SVM(kernel='polynomial', C=C, degree=parameter['degree'],
                   solver=solver, decision_function_shape=decision_function_shape)
    else:
        raise ValueError("Unsupported kernel type.")
    
def cross_validation(X, y, kernel_type, parameter_grid, C_values, solver ,k=3,decision_function_shape=None):
    """
    Perform k-fold cross-validation to select the best hyperparameters.

    Args:
        X: Data (n_samples, n_features)
        y: Labels (-1 or +1)
        kernel_type: 'gaussian' or 'polynomial'
        parameter_grid: List of dicts with kernel parameters (gamma or degree)
        C_values: List of C values to test
        k: Number of folds
        decision_function_shape: ovo or ova for Muticlass Classification (default value is set to None for the Binary Classification)

    Returns:
        best_parameters: Dictionary with best C and kernel parameters
        best_accuracy: Best average accuracy obtained
    """
    best_accuracy = 0.0
    best_parameters = {}
    all_results = []
    for C in C_values:
        for parameter in parameter_grid:

            accuracy_list = []
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X,y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                # Create SVM instance with current parameters using build_svm
                svm = build_svm(kernel_type=kernel_type,C=C,parameter=parameter,solver=solver,decision_function_shape=decision_function_shape)
                
                try:
                    # Train the model
                    svm.fit(X_train, y_train)
                    # Test the model
                    y_val_pred=svm.predict(X_val)
                    acc = svm.score(y_val_pred, y_val)
                    accuracy_list.append(acc)

                except Exception as e:
                    print(f"Error with C={C}, parameter={parameter}: {e}")
                    continue

            # Compute average accuracy over k folds
            if accuracy_list:
                average_accuracy = np.mean(accuracy_list)

                config_result = {'kernel': kernel_type, 'C': C, **parameter, 'accuracy': round(average_accuracy,3)}
                all_results.append(config_result)

                if average_accuracy > best_accuracy:
                    best_accuracy = average_accuracy
                    best_parameters = {'C': C, **parameter}

    return best_parameters, best_accuracy,all_results

def select_best_configuration(X:np.ndarray, y:np.ndarray,solver: Optional[Literal["cvxopt", "mvp"]] = 'cvxopt',k:int = 3,decision_function_shape: Optional[Literal["ova", "ovo"]] = None, 
                              kernel_configurations: list = [], C_values:list = []):
    """Compare different kernels and select the best one using cross-validation.

        Args:
            X: Data (n_samples, n_features)
            y: Labels (-1 or +1)
            solver: CVXOPT or MVP
            decision_function_shape: ovo or ova for Muticlass Classification (default value is set to None for the Binary Classification)
            kernel_configurations: different configurations of gaussian and polynomial kernel
            C_values: List of C values to test
            k: Number of folds

        Returns:
            best_overall: Dictionary with best configuration for each Kernel
            results: Dictionary with best results for each kernel
    """

    if not (kernel_configurations and C_values):
        raise ValueError("Both `configuration` and `C_values` must be provided and non-empty.")
    
    best_overall_accuracy = 0
    best_overall = {}
    results, all_results = [], []

    for config in kernel_configurations:
        print(f"Testing kernel: {config['name']}")
        
        # Use cross-validation with SVM
        best_parameters, best_accuracy, kernel_results  = cross_validation(
            X, y, 
            solver = solver,
            k=k,
            kernel_type = config["name"], 
            parameter_grid = config["param_grid"], 
            C_values = C_values,
            decision_function_shape = decision_function_shape)
        
        print(f"Best {config['name']} Kernel Accuracy: {best_accuracy:.3f}")
        all_results.extend(kernel_results) 
        
        results.append({"kernel": config["name"],"accuracy": best_accuracy,"parameters": best_parameters})
        
        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall = {"kernel": config["name"],"parameters": best_parameters}
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("accuracy", ascending=False)
    display(results_df)

    return best_overall, results

def plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred, mode: Optional[Literal["binary", "multiclass"]] = 'binary',figsize: tuple = (18, 6),
                          class_names: Optional[List[str]] = None):
    if mode=="binary":
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        Train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(Train_confusion_matrix, annot=True, fmt='.0f', cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'], linewidths=1, linecolor="white")
        plt.title("Traning Confusion Matrix",fontsize=14, weight="bold")

        plt.subplot(1, 2, 2)
        Test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(Test_confusion_matrix, annot=True, fmt='.0f', cmap=sns.cubehelix_palette(start=.2, rot=-.1, as_cmap=True),
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'], linewidths=1, linecolor="white")
        plt.title("Test confusion Matrix",fontsize=14, weight="bold")
        plt.show()

    elif mode =="multiclass":
        labels = np.unique(np.concatenate((y_train, y_test))) if class_names is None else class_names

        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        Train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(Train_confusion_matrix, annot=True, fmt='.0f', cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                    xticklabels=labels, yticklabels=labels , linewidths=1, linecolor="white")
        plt.title("Traning Confusion Matrix",fontsize=14, weight="bold")

        plt.subplot(1, 2, 2)
        Test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(Test_confusion_matrix, annot=True, fmt='.0f', cmap=sns.cubehelix_palette(start=.2, rot=-.1, as_cmap=True),
                    xticklabels=labels, yticklabels=labels , linewidths=1, linecolor="white")
        plt.title("Test confusion Matrix",fontsize=14, weight="bold")
        plt.show()
    
def plot_decision_boundary(svm, X: np.ndarray, y: np.ndarray, title="SVM Decision Boundary", figsize=(10,6)):

    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(grid).reshape(xx.shape)
    
    plt.figure(figsize=figsize)

    colors = list(plt.cm.Set1(np.arange(len(np.unique(y)))))
    colors[0], colors[1] = colors[1], colors[0]
    custom_cmap = ListedColormap(colors)

    plt.contourf(xx, yy, Z, levels=len(np.unique(y)), cmap=custom_cmap, alpha=0.6)

    for i, classes in enumerate(np.unique(y)):
        plt.scatter(X[y==classes, 0], X[y==classes, 1], 
                   label=f'Class {i+1}', edgecolors='k', alpha=0.8, color=colors[i])

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

