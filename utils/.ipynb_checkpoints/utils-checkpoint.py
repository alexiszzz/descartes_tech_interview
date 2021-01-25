import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def to_nominal(row: str, labels: list) -> float:
    """Replace str labels with numerical labels"""
    result = labels.index(row)
    return result


def to_nominal_df(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create df with numerical labels and store the matching strategy"""
    df_num = df['INDEX'].to_frame()
    labels_dict = {}
    for col in columns:
        labels = df[col].unique().tolist()
        df_num[col] = df[col].apply(lambda x: to_nominal(x, labels))
        labels_dict[col] = labels
    return df_num, labels_dict


def to_nominal_df_test(df: pd.DataFrame, labels_dict: dict) -> pd.DataFrame:
    """Create df with numerical labels for test sets
    based on same matching strategy"""
    df_num = df['INDEX'].to_frame()
    columns = list(labels_dict.keys())
    for col in columns:
        labels = labels_dict[col]
        df_num[col] = df[col].apply(lambda x: to_nominal(x, labels))
    return df_num


def to_numerical(row: str) -> float:
    """Transform str money value into float"""
    if type(row) == str:
        result = row.replace('$', '')
        result = result.replace(',', '')
        result = float(result)
    else:
        result = row
    return result


def to_numerical_df(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create df with numerical money values"""
    df_num = df['INDEX'].to_frame()
    for col in columns:
        df_num[col] = df[col].apply(to_numerical)
    return df_num


def plot_importance(importance: np.array, names: list, model_type: str) -> None:
    """Plot feature importance"""
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    plt.figure(figsize=(7, 5))
    plt.rc('ytick', labelsize=10)
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + ' feature importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')


def plot_corr_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(23, 20))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1,
                          annot=True, fmt='.2f', cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')