import numpy as np 
import pandas as pd
import seaborn as sns
sns.reset_defaults
sns.set_style(style='darkgrid')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
font = {'size'   : 16}
plt.rc('font', **font)
plt.ion()
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["figure.figsize"] = (20.0, 10.0)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)

# quantity clicked from each country
def qty_by_feature(df, columns):
    '''
    Prints a series of bar charts displaying the qunatity of users who clicked on a link in a marketing email by feature. 
    Useful for identifying which features lead to a user clicking on the link.

    Parameters:
    ----------
    input {dataframe, list}: dataframe of users and click data, list of features to plot on
    output {plots}: a series of barplots representing the quantity of users who clicked in each features category
    '''

    for column in columns:
        percents = {}
        for item in df[column].unique():
            pct = df[df[column] == item]['clicked'].sum() / df[df[column] == item].shape[0]
            percents[item] = pct
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        ax[0].bar(x=df[column].value_counts().index, height=df[column].value_counts().values, color="orange")
        ax[0].set_title("Users who Received Email by {}".format(column))
        ax[0].set_xlabel("Value")
        ax[0].set_ylabel("Qty")
        ax[1].bar(x=df[df['clicked'] == 1][column].value_counts().index, height=df[df['clicked'] == 1][column].value_counts().values, color="blue", alpha=0.8)
        ax[1].set_title("Users Who Clicked by {}".format(column))
        ax[1].set_xlabel("Value")
        ax[1].set_ylabel("Clicked")
        plt.tight_layout()
        plt.show()
        

def pct_by_feature(df, columns):
    '''
    Prints a series of bar charts displaying the percents of users who clicked on a link in a marketing email by feature. 
    Useful for identifying which features lead to a user clicking on the link.

    Parameters:
    ----------
    input {dataframe, list}: dataframe of users and click data, list of features to plot on
    output {plots}: a series of barplots representing the percentage of users who clicked in each features category
    '''
    for column in columns:
        percents = {}
        for item in df[column].unique():
            pct = df[df[column] == item]['clicked'].sum() / df[df[column] == item].shape[0]
            percents[item] = pct
        plt.figure(figsize=(8,4))
        plt.bar(x=percents.keys(), height=percents.values(), color="blue", alpha=0.7)
        plt.title("Pct Who Clicked by {}".format(column))
        plt.xlabel("Value")
        plt.ylabel("Qty")
        plt.tight_layout()
        plt.show()
    
    
####################################################################
if __name__ == "__main__":

    df = pd.read_csv('data/joined_data.csv')

    cols = ['email_text', 'email_version', 'hour', 'weekday',
    'user_country', 'user_past_purchases']

    pct_by_feature(df, cols)
