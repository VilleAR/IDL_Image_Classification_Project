import os 
import pandas as pd 


ids=[]
for i in range(20001,25001):
    s="im"+str(i)+".jpg"
    ids.append(s)

df=pd.read_csv('preds.csv', sep=',')
df.insert(loc=0, column='ids', value=ids)

df.to_csv(r'FINALpredictions.csv',index=False)
