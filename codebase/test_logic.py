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

# make random np array
array = np.random.rand(2,2)
k = np.stack(array)
print(array)
print(k)
print(np.equal(array,k))

