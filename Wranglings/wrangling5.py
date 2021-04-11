import os 

with open('labels.txt') as f:
    lines=f.readlines()
    i=0
    trains=[]
    vals=[]
    while i<15000:
        trains.append(lines[i])
        i+=1
    while i<20000:
        vals.append(lines[i])
        i-=-1
with open('labelstrain.txt', 'w') as f:
    for item in trains:
        f.write("%s" % item)

with open('labelsval.txt', 'w') as f:
    for item in vals:
        f.write("%s" % item)
