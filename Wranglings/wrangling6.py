import os
import shutil


zz=[]
with open('../labels.txt', 'r') as f:
    lines=f.readlines()
    for i,l in enumerate(lines):
        w=False
        s=l.split(' ')
        for q in s:
            if q=='1':
                w=True
        if w==False:
            zz.append(i+1)

with open('suspects.txt', 'w') as f:
    for item in zz:
        f.write("%s\n" % item)

    