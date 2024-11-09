import os
# path_1 = '/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Train/1ADC_ss'
# path_2 = '/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Train/2Z_ADC'

# paths = [path_1, path_2]

# combo = [sorted([i for i in os.listdir(path) if i.endswith('.mha')]) for path in paths]
# # for path in paths:
# #     combo.append([i for i in os.listdir(path)])

# print(*list(zip(*combo))[:2],sep="\n")
# # print(*list(zip(combo)[:10],sep="\n"))

# print(len(combo))
# print(len(combo[0]))
# print(len(combo[1]))

# print(*sorted(combo[0])[:10],sep="\n")
# print()
# print(*sorted(combo[1])[:10],sep="\n")

import numpy as np

a = np.load('/Users/amograo/Research_Projects/DL_HIE_2024/test_saving/MGHNICU_010-VISIT_01-ADC_ss_slice_0.npy')
print(a.shape)
print(np.stack(a).shape)
print(np.stack(a,axis=0).shape)
print(np.equal(a,np.stack(a,axis=0)).all())
print(np.equal(np.stack(a),np.stack(a,axis=0)).all())
print(np.expand_dims(a,axis=0).shape)
# make random np array
array = np.random.rand(2,2)
k = np.stack(array,axis=-1)
print(array,array.shape)
print(k,k.shape)
print(np.equal(array,k))
print(np.equal(array.transpose(1,0),k))
# stack the array such that shape is 2,2,3


print(np.stack([array,array,array],axis=-1).shape)
j = np.expand_dims(array,axis=-1)
print(j.shape)
print(j.transpose(2, 0, 1).shape)
print(np.expand_dims(array,axis=0).shape)
