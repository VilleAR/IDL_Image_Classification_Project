import os 
from natsort import natsorted 

directory=r'test'
zz=[]
for name in os.listdir(directory):
    zz.append(name)
zz=natsorted(zz)
with open('imsval.txt', 'w') as f:
    for item in zz:
        f.write("%s\n" % item)