import math
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif

    
def my_continuous_plot(data, variables_list, n_cols, length):
    n_rows = math.ceil(len(variables_list)/n_cols)*2
    fig = plt.figure(figsize=(length*n_cols, 4*n_rows))
    i,j = 1,2

    for variable in (variables_list):
        ax = fig.add_subplot(n_rows,n_cols,i)    
        sns.histplot(data=data[variable])
        ax.set_title(variable)
        i+=2
    
    for variable in (variables_list):
        ax = fig.add_subplot(n_rows,n_cols,j)
        sns.boxplot(data=data[variable], orient='h')
        ax.set_title(variable)
        j+=2

    plt.tight_layout()
    
     
def my_barplot(X, Y, length, width): 
    fig = plt.figure(figsize=(length, width))
    sns.barplot(x=X, y=Y, palette="Paired")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()

    
def my_count_plot (data, variable, length, width, tit_plot, tit_y, tit_color, tit_fontsize):
    fig = plt.figure(figsize=(length, width))
    sns.countplot(x=data[variable])
    plt.ylabel(tit_y)
    plt.title(tit_plot,color = tit_color,fontsize=tit_fontsize)

    
def my_pair_plot(data, size_graph):
    sns.pairplot(data, size = size_graph)
    plt.show()
    

def select_features(X_train, y_train, X_test, score, n_column_sel):
    fs = SelectKBest(score_func=score, k=n_column_sel)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def conf_matrix(data, tit_plot, tit_x, tit_y):
    ax = sns.heatmap(data/np.sum(data), annot=True, 
            fmt='.2%', cmap='Blues')
    ax.set_title(tit_plot)
    ax.set_xlabel(tit_x)
    ax.set_ylabel(tit_y)
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    plt.show()
    
def my_barplot_pct(data, X, Y, length, width, tit_plot, tit_color, tit_fontsize): 
    ax = sns.barplot(X, Y,hue=None, data=data,errwidth=0)
    plt.title(tit_plot,color = tit_color,fontsize=tit_fontsize)
    for i in ax.containers:
        ax.bar_label(i,)    