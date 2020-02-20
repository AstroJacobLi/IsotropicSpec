import tqdm
import h5py
import numpy as np
import sys
# python load_csv_into_h5.py xxx.csv
fn = sys.argv[1]
f = open(fn,'r')
title = f.readline()
cnt = 0
data = np.zeros((573474,2600), dtype=np.float32)
label = np.zeros((573474,), dtype=np.int)
ids = []
for i in tqdm.tqdm(range(573474)):
    line = f.readline()
    points = line.split(',')
    if 'FE' in line:
        continue
    if not len(points) == 2602:
        print('Error in ', i)
        import IPython
        IPython.embed()
    data[i,:] = np.array(points[:-2], dtype=np.float32)
    if points[-2] == 'star':
        label[i] = 0
    elif points[-2] == 'galaxy':
        label[i] = 1
    elif points[-2] == 'qso':
        label[i] = 2
    else:
        print(points[-2])
        import IPython
        IPython.embed()
    ids.append(points[-1])
    cnt += 1
print(cnt)
hf = h5py.File('./train_v1.h5','w')
hf['data'] = data[:cnt,...]
hf['label'] = label[:cnt]
hf['ids']  = np.array(ids, dtype='S')
hf.close()
import IPython
IPython.embed()
