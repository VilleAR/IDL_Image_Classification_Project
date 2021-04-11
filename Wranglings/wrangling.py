import shutil
import os

directory=r'annotations'
for im in images(directory):
    print(im)


for name in os.listdir(directory):
    s=directory+'/'+name
    target='data/alldata/'+name[:-4]
    print(target)
    with open(s) as f:
        lines=f.readlines()
        zz=[]
        for r in lines:
            zz.append(int(r))
        zz=list(zz)
        for e in zz:
            t='im'+str(e)+'.jpg'
            shutil.copy('images/'+t, target+'/'+t)
            #print(target+t)
