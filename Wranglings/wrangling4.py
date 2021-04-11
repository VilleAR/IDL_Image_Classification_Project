import os 

zz=[]

for i in range(1, 15001):
    s='im'
    s+=str(i)
    s+='.jpg'
    zz.append(s)
xx=[]
for i in range(15001, 20001):
    s='im'
    s+=str(i)
    s+='.jpg'
    xx.append(s)

with open('imstrain.txt', 'w') as f:
    for item in zz:
        f.write("%s\n" % item)

with open('imsval.txt', 'w') as f:
    for item in xx:
        f.write("%s\n" % item)