import pandas as pd

users_df = pd.read_excel('./weibo_user2.xlsx')
users_df = users_df[users_df.fans != 0]
users_df['f_f'] = 0.0
for index, row in users_df.iterrows():
    users_df['f_f'][index] = row.followers / row.fans
users_df.to_excel('./weibo_user2-f-f.xlsx', index=False)
