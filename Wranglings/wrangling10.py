import os 

arr=[]
with open('src/suspects.txt') as f:
    lines=f.readlines()
    for l in lines:
        arr.append(int(l))
labelstrain=[]
with open('labelsval.txt') as f:
    lines=f.readlines()
    labelstrain=lines
badindexes=[]
for i in arr:
    if i>20000:
        break 
    elif i>15000:
        badindexes.append(i-1)

badindexes=list(reversed(badindexes))
print(badindexes[0:10])
for b in badindexes:
    labelstrain.pop(b-15000)
#print(labelstrain)

with open('labelsval.txt', 'w') as f:
    for item in labelstrain:
        f.write("%s" % item)

