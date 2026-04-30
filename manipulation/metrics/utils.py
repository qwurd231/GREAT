# from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import pingouin as pg 

# methods: pearson, spearman
def partial_corr(data, x_list, y, covar, method="pearson"):
    results = {}
    for var in x_list:
        result = pg.partial_corr(data=data, x=var, y=y, covar=covar, method=method)
        results[var] = result
    print("Partial Correlation Results:")
    for var, res in results.items():
        print(f"{var} except {covar} for {y}")
        print(f"{res['r'].values[0]}, p-value: {res['p-val'].values[0]}")

# print the correlation
def corr(pd,name1,name2):
    array1 = pd[name1]
    array2 = pd[name2]
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    print(f'{name1} vs {name2} pearson & spearman')
    pearson_corr, pearson_p_value = pearsonr(array1, array2)
    print(f'{pearson_corr}, p-value: {pearson_p_value}')
    spearman_corr, spearman_p_value = spearmanr(array1, array2)
    print(f'{spearman_corr}, p-value: {spearman_p_value}')

def plot_correlation(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    # from scipy.stats import pearsonr
    # print(data.columns)
    # y = data['mt']
    # x = data.drop(columns=['mt'])
    # corr, p_value = spearmanr(x=x, y=y, axis=0)
    corr, p_value = spearmanr(data, axis=0)
    corr_matrix = pd.DataFrame(corr, columns=data.columns, index=data.columns)
    p_value_matrix = pd.DataFrame(p_value, columns=data.columns, index=data.columns)

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    mask = (p_value_matrix == 0)
    sns.heatmap(corr_matrix, annot=True, cmap='Blues_r', fmt='.4f', vmin=-1, vmax=1, ax=ax[0], center=0, mask=mask)
    sns.heatmap(p_value_matrix, annot=True, cmap='Blues_r', fmt='.4f', vmin=0, vmax=1, ax=ax[1], mask=mask)
    ax[0].set_title("pearson Correlation Coefficients")
    ax[1].set_title("P-values")

    plt.tight_layout()
    plt.show()
    
####################
# not in use
def person_corr(array1, array2):    
    pearson_corr, pearson_p_value = pearsonr(array1, array2)
    print(f'Pearson correlation coefficient: {pearson_corr}, p-value: {pearson_p_value}')
    
# not in use
def _plot_correlation(data):
    # corr, p_value = spearmanr(data, axis=0)
    corr, p_value = pearsonr(data, axis=0)
    corr_matrix = pd.DataFrame(corr, columns=data.columns, index=data.columns)
    p_value_matrix = pd.DataFrame(p_value, columns=data.columns, index=data.columns)

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    mask = (p_value_matrix == 0)
    sns.heatmap(corr_matrix, annot=True, cmap='Blues_r', fmt='.4f', vmin=-1, vmax=1, ax=ax[0], center=0, mask=mask)
    sns.heatmap(p_value_matrix, annot=True, cmap='Blues_r', fmt='.4f', vmin=0, vmax=1, ax=ax[1], mask=mask)
    ax[0].set_title("Pearson Correlation Coefficients")
    ax[1].set_title("P-values")

    plt.tight_layout()
    plt.show()
    
def plot_correlation_matrix(data):
    # corr, p_value = spearmanr(data, axis=0)
    corr, p_value = pearsonr(data, axis=0)
    corr_matrix = pd.DataFrame(corr, columns=data.columns, index=data.columns)
    p_value_matrix = pd.DataFrame(p_value, columns=data.columns, index=data.columns)

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    mask = (p_value_matrix == 0)
    sns.heatmap(corr_matrix, annot=True, cmap='Blues_r', fmt='.4f', vmin=-1, vmax=1, ax=ax[0], center=0, mask=mask)
    sns.heatmap(p_value_matrix, annot=True, cmap='Blues_r', fmt='.4f', vmin=0, vmax=1, ax=ax[1], mask=mask)
    ax[0].set_title("Pearson Correlation Coefficients")
    ax[1].set_title("P-values")

    plt.tight_layout()
    plt.show()
    
def plot_corr(X1, X2, X3):
    
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

    corr_matrix = data.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    X = data[['X1', 'X2']]  
    y = data['X3']         

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"R^2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    residuals = y - y_pred
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True Values vs Predicted Values")
    plt.show()

    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()

    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    sm.qqplot(residuals, line='45')
    plt.title("Q-Q Plot of Residuals")
    plt.show()
    
    
