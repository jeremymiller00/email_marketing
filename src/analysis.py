import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
    
# read in data
df = pd.read_csv('data/email_table.csv')
open_df = pd.read_csv('data/email_opened_table.csv')
click_df = pd.read_csv('data/link_clicked_table.csv')

# check that all email ids are unique
print(df['email_id'].nunique() == df.shape[0])

# merge dataframes
col1 = np.ones(open_df.shape[0])
col2 = np.ones(click_df.shape[0])

open_df['opened'] = col1
click_df['clicked'] = col2

joined_df = pd.merge(df, open_df, how='outer', on='email_id')
joined_df = pd.merge(joined_df, click_df, how='outer', on='email_id')
joined_df.fillna(0, inplace=True)

# check that merged values are correct
print(joined_df['opened'].sum() == open_df.shape[0])
print(joined_df['clicked'].sum() == click_df.shape[0])

pct_opened = joined_df['opened'].sum() / joined_df.shape[0]
pct_clicked = joined_df['clicked'].sum() / joined_df.shape[0]
print( "Percentage of people who opened the email: {}%".format(pct_opened*100) )
print( "Percentage of people who clicked on the email: {}%".format(pct_clicked*100) )

joined_df.to_csv('data/joined_data.csv', index=False)