import os
import pandas as pd
import json
from datetime import datetime as dt
import re
from itertools import compress
import ast

data_path = os.path.join('D:/', 'data', 'horesta')
filename = 'horesta_posts_2021-03-25.json'
outname = 'horesta_posts_20210325.csv'

# Loading data
path = os.path.join(data_path, filename)

with open(path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame.from_records(data)
    
df_select = df.loc[:, df.columns != 'html']
df_select['text'] = df_select['text'].apply(lambda string: string.replace("\n", ""))
df_select.to_csv(os.path.join(data_path, outname), index = False)


# Exporting texts

df_export = df.copy()
df_export['publish_date'] = df_export['publish_date'].str.replace(' ', '')
df_export['text'] = df_export['title'] + df_export['text']

outdir = os.path.join(data_path, 'txt')
old_postdate = ""
counter = 1

for postindex in list(df_export.index):
    postdate = df_export.loc[postindex, 'publish_date']
    
    year = postdate[6:10]
    month = postdate[3:5]
    day = postdate[0:2]
    
    postdate = year + "-" + month + "-" + day
    
    indexstr = str(postindex).rjust(4, '0')
        
    filename = "horesta_{date}_{index}.txt".format(date = postdate, index = indexstr)
    
    old_postdate = postdate

    outpath = os.path.join(outdir, filename)
    
    if os.path.isfile(outpath):
        continue
    
    with open(outpath, 'w', encoding = "utf-8") as f:
        f.write(str(df_export.loc[postindex, 'text']))