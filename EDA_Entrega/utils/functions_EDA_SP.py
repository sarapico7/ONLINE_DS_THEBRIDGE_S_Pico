import matplotlib.pyplot as plt
import numpy as np
from omniplot  import plot as op
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 
def boxplot_with_outliers(df, columns):
    """
    Generate boxplots with possible outliers for numeric columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame.
    - columns: List of numeric column names.

    Example of usage:
    boxplot_with_outliers(df, columns=["Column1", "Column2"])
    """
    # Create a figure with subplots for each column
    num_columns = len(columns)
    fig, axes = plt.subplots(1, num_columns, figsize=(10, 4))

    # Iterate over columns and draw boxplots with vertical orientation
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'Boxplot - {col}')

    plt.tight_layout()
    plt.show()

def categorical_distribution(df, columns, relative=False, values=False, giro = 45):
    num_columnas = len(columns)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(10, 4 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columns):
        ax = axes[i]
        if relative:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='Set2', hue = serie.index, legend = False)
            ax.set_ylabel('Relative frequency')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='Set2', hue=serie.index, legend=False)
            ax.set_ylabel('Frecuency')

        ax.set_title(f'{col} distribution')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=giro)

        if values:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def boxplots(df):
    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=np.number).columns

    # Calculate the number of rows and columns needed for subplots
    num_columns = len(numerical_columns)
    num_rows = (num_columns // 4) + (num_columns % 4 > 0)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(15, 4 * num_rows))
    fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Create box plots for each numerical column with outliers
    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i], color='skyblue', width=0.5, fliersize=5)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Values')

    # Hide the empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Show the plot
    plt.show()
    

 
def scatter_plots_aggregated(df, col_num1="age_2016"):
    """
    Generates overlaid scatter plots of a numerical column (default "age_2016")
    compared with all other numerical columns in the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    col_num1 (str): Name of the first numerical column for the X-axis (default is "age_2016").
    """
    # Filter numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Configuration to enhance the aesthetics of the plot
    sns.set(style="darkgrid")

    # Calculate the correct number of subplots
    num_subplots = len(numeric_cols) - 1  # Exclude "age_2016"
    num_rows = int(np.ceil(num_subplots / 2))
    num_cols = 2

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    fig.tight_layout(pad=3.0)

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    for i, col_num2 in enumerate(numeric_cols):
        if col_num2 != col_num1:
            # Check if the subplot index is within bounds
            if i < num_subplots:
                # Drop rows with NaN in both columns
                df_subset = df[[col_num1, col_num2]].dropna()

                # Use seaborn to generate overlaid scatter plots
                scatter = sns.scatterplot(x=col_num1, y=col_num2, data=df_subset, ax=axes[i], label=col_num2)

                # Calculate regression and correlation
                reg_model = LinearRegression()
                X = df_subset[[col_num1]]
                y = df_subset[col_num2]
                reg_model.fit(X, y)
                y_pred = reg_model.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                # Draw the regression line on the plot
                axes[i].plot(X, y_pred, color='black', linewidth=2)

                # Add title and labels
                axes[i].set_title(f'Scatter Plot: {col_num1} vs {col_num2}')
                axes[i].set_xlabel(col_num1)
                axes[i].set_ylabel(col_num2)

                # Show legend with regression and correlation information
                legend_text = f'Regression: y = {reg_model.coef_[0]:.2f}x + {reg_model.intercept_:.2f}\nCorrelation: {df_subset.corr().iloc[0, 1]:.2f}'
                axes[i].legend(title=legend_text)

    # Show the subplots
    plt.show()



def calculate_correlation_regression(df, col_num1="age_2016"):
    """
    Calculates correlations and regression parameters for a numerical column
    compared with all other numerical columns in the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    col_num1 (str): Name of the first numerical column for the X-axis (default is "age_2016").

    Returns:
    pd.DataFrame: DataFrame containing correlations, p-values, and regression parameters.
    """
    # Filter numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Create a DataFrame to store correlations and regression parameters
    results_df = pd.DataFrame(columns=['Column1', 'Column2', 'Correlation', 'P_Value', 'Regression_Coeff', 'Intercept'])

    for col_num2 in numeric_cols:
        if col_num2 != col_num1:
            # Drop rows with NaN values in the selected columns
            df_filtered = df[[col_num1, col_num2]].dropna()

            # Check if there are still enough data points for analysis
            if len(df_filtered) > 1:
                # Calculate regression and correlation
                reg_model = LinearRegression()
                X = df_filtered[[col_num1]]
                y = df_filtered[col_num2]
                reg_model.fit(X, y)
                y_pred = reg_model.predict(X)

                # Calculate correlation and p-value
                correlation, p_value = pearsonr(X.squeeze(), y)

                # Store results in the DataFrame
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Column1': [col_num1],
                    'Column2': [col_num2],
                    'Correlation': [correlation],
                    'P_Value': [p_value],
                    'Regression_Coeff': [reg_model.coef_[0]],
                    'Intercept': [reg_model.intercept_]
                })], ignore_index=True)

    return results_df




def scatter_plots_aggregated_with_categorical(df, col_num1="age_2016", col_cat=None):
    """
    Generates overlaid scatter plots of a numerical column (default "age_2016")
    compared with all other numerical columns in the DataFrame, grouped by a categorical column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    col_num1 (str): Name of the first numerical column for the X-axis (default is "age_2016").
    col_cat (str): Name of the categorical column for grouping (default is None).
    """
    # Filter numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Filter categorical column
    if col_cat:
        df = df[df[col_cat].notnull()]  # Drop rows with null values in the selected categorical column

        # Iterate over unique categories in the categorical column
        for category in df[col_cat].unique():
            # Filter the DataFrame for the current category
            df_category = df[df[col_cat] == category]

            # Configuration to enhance the aesthetics of the plot
            sns.set(style="darkgrid")
            sns.set_palette("Set3")

            # Calculate the correct number of subplots
            num_subplots = len(numeric_cols) - 1  # Exclude "age_2016"
            num_rows = int(np.ceil(num_subplots / 2))
            num_cols = 2

            # Create subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))
            fig.tight_layout(pad=3.0)

            # Flatten the axes array to simplify indexing
            axes = axes.flatten()

            for i, col_num2 in enumerate(numeric_cols):
                if col_num2 != col_num1:
                    # Check if the subplot index is within bounds
                    if i < num_subplots:
                        # Drop rows with NaN values in the selected columns
                        df_subset = df_category[[col_num1, col_num2]].dropna()

                        # Check if there are still enough data points for analysis
                        if len(df_subset) > 1:
                            # Use seaborn to generate overlaid scatter plots
                            scatter = sns.scatterplot(x=col_num1, y=col_num2, data=df_subset, ax=axes[i], label=category)

                            # Calculate regression and correlation
                            reg_model = LinearRegression()
                            X = df_subset[[col_num1]]
                            y = df_subset[col_num2]
                            reg_model.fit(X, y)
                            y_pred = reg_model.predict(X)

                            # Draw the regression line on the plot
                            axes[i].plot(X, y_pred, color='black', linewidth=2)

                            # Add title and labels
                            axes[i].set_title(f'Scatter Plot: {col_num1} vs {col_num2} ({category})')
                            axes[i].set_xlabel(col_num1)
                            axes[i].set_ylabel(col_num2)

                            # Show legend with regression and correlation information
                            legend_text = f'Regression: y = {reg_model.coef_[0]:.2f}x + {reg_model.intercept_:.2f}\nCorrelation: {df_subset.corr().iloc[0, 1]:.2f}'
                            axes[i].legend(title=legend_text, loc='upper left')

            # Show the subplots for the current category
            plt.show()



def calculate_correlation_regression_with_categorical(df, col_num1="age_2016", col_cat=None):
    """
    Calculates correlations, p-values, and regression parameters for a numerical column
    compared with all other numerical columns in the DataFrame, grouped by a categorical column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    col_num1 (str): Name of the first numerical column for the X-axis (default is "age_2016").
    col_cat (str): Name of the categorical column for grouping (default is None).

    Returns:
    pd.DataFrame: DataFrame containing correlations, p-values, and regression parameters.
    """
    # Filter numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Filter categorical column
    if col_cat:
        df = df[df[col_cat].notnull()]  # Drop rows with null values in the selected categorical column

        # Create a DataFrame to store correlations and regression parameters
        results_df = pd.DataFrame(columns=['Column1', 'Column2', 'Category', 'Correlation', 'P_Value', 'Regression_Coeff', 'Intercept'])

        # Iterate over unique categories in the categorical column
        for category in df[col_cat].unique():
            # Filter the DataFrame for the current category
            df_category = df[df[col_cat] == category]

            for col_num2 in numeric_cols:
                if col_num2 != col_num1:
                    # Calculate regression and correlation
                    reg_model = LinearRegression()
                    X = df_category[[col_num1]]
                    y = df_category[col_num2]

                    # Drop rows with NaN values in the selected columns
                    not_nan_mask = ~np.isnan(X.values.flatten()) & ~np.isnan(y.values)
                    X = X[not_nan_mask].values.reshape(-1, 1)
                    y = y[not_nan_mask]

                    # Check if there are still enough data points for analysis
                    if len(X) > 1:
                        reg_model.fit(X, y)
                        y_pred = reg_model.predict(X)

                        # Calculate correlation and p-value
                        correlation, p_value = pearsonr(X.squeeze(), y)

                        # Store results in the DataFrame
                        results_df = pd.concat([results_df, pd.DataFrame({
                            'Column1': [col_num1],
                            'Column2': [col_num2],
                            'Category': [category],
                            'Correlation': [correlation],
                            'P_Value': [p_value],
                            'Regression_Coeff': [reg_model.coef_[0]],
                            'Intercept': [reg_model.intercept_]
                        })], ignore_index=True)

        return results_df



def filter_and_extract_values(df, column_name, threshold, corresponding_column):
    """
    Filters values in a specified column based on a threshold and extracts corresponding values from another column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to filter.
    threshold (float): Threshold value for filtering.
    corresponding_column (str): Name of the corresponding column.

    Returns:
    pd.DataFrame: New DataFrame containing filtered values and corresponding values from another column.
    """
    condition = (df[column_name] > threshold) | (df[column_name] < -threshold)
    filtered_df = df[condition]

    if not filtered_df.empty:
        result_df = pd.DataFrame({
            corresponding_column: filtered_df[corresponding_column].values,
            column_name: filtered_df[column_name].values
        })
        return result_df
    else:
        return pd.DataFrame(columns=[corresponding_column, column_name])


def plot_all_variables_across_groups(df):
    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        variable_name = row['Column2']
        variable_data = row['Correlation':].astype(float)

        # Configuration to enhance the aesthetics of the plot
        sns.set(style="darkgrid")
        sns.set_palette("husl")

        # Create a line plot for the variable
        
        plt.figure(figsize=(12, 6))
        plot = sns.lineplot(x=df.columns[1:], y=variable_data.values, marker='o', label=variable_name)

        # Customize the plot
        plt.title(f'Line Plot for {variable_name} Across Groups')
        plt.xlabel('Groups')
        plt.ylabel('Values')
        plt.legend()

        # Rotate x-axis labels for better readability
        plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha='right')

        # Show the plot
        plt.show()



def find_max_min(row):
    column2 = row['Column2']
    # Exclude the 'Column2' column to find the max and min
    row_values = row.drop('Column2')
    
    # Drop NaN values from the row
    row_values = row_values.dropna()
    
    # Check if there are any values left in the row
    if row_values.empty:
        return pd.Series({
            'Column2': column2,
            'Max_Correlation': None,
            'Max_Correlation_Column': None,
            'Min_Correlation': None,
            'Min_Correlation_Column': None
        })
    
    # Find the maximum and its corresponding column title
    max_column = row_values.idxmax()
    max_value = row_values[max_column]
    
    # Find the minimum and its corresponding column title
    min_column = row_values.idxmin()
    min_value = row_values[min_column]
    
    return pd.Series({
        'Column2': column2,
        'Max_Correlation': max_value,
        'Max_Correlation_Column': max_column,
        'Min_Correlation': min_value,
        'Min_Correlation_Column': min_column
    })






