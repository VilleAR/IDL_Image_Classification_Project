import os
import shutil


directory=r'annotations'
i=1
tensors=[]
log_interval=1000
zz=[]
while i<=20000:
    hits=[]
    tens=''
    if i%log_interval==0:
            print(i)
    for name in os.listdir(directory):     
        s=directory+'/'+name
        with open(s) as f:
            lines=f.readlines()
            ints=[]
            for b in lines:
                ints.append(int(b))
            ints.sort()        
            found=False
            for r in ints:
                if int(r)==i:                
                    tens+='1 '
                    found=True
                    #hits.append(name)
                    break 
                elif int(r)>i:
                    break
            if found==False:
                tens+='0 '
    i+=1
    tensors.append(tens)
#print(tensors[0:10])
with open('labels.txt', 'w') as f:
    for item in tensors:
        f.write("%s\n" % item)

