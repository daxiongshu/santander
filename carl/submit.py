import os
import sys
import glob

def get_last(name='run.log'):
    with open(name) as f:
        for line in f:
            pass
    tag = line.split(',')
    tag = '%s_cv_%s'%(tag[0],tag[1])
    print(tag)
    return glob.glob('backup/%s/*sub*gz'%tag)[0]

if len(sys.argv)<2:
    path = get_last()
else:
    path = sys.argv[1]
xx = path.split('/')
filex = glob.glob('/'.join(xx[:2])+'/*sub*gz')[0]
#print(filex)
mes = xx[1]
#print(mes)
cmd = 'kaggle competitions submit -c santander-customer-transaction-prediction -f %s -m %s'%(filex,mes)
print(cmd)
os.system(cmd)
