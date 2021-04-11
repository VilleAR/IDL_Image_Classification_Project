import os
import shutil as shutil
import re as re 

directory=r'../train'

arr=[]
with open('suspects.txt', 'r') as f:
    lines=f.readlines()
    for l in lines:
        arr.append(int(l))

gex=[]
for name in os.listdir(directory):
    n=name.replace('im','')
    n=n.replace('.jpg','')
    n=int(n)
    w=False
    for a in arr:
        if a==n:
            w=True
    if w:
        gex.append(name)

with open('gex1.txt', 'w') as f:
    for item in gex:
        f.write("%s\n" % item)
