'''
this code is to create a full data set including 6049758 samples
in a single sample, it includes 31 features and 1 label as the input and output, respectively.
the output file is *.npy file
'''

import numpy as np
import FlowCal
import os
import pathlib
import pandas as pd

# get a list of all specified features
df = pd.read_excel(r'EU_marker_channel_mapping.xlsx')
channels = []
for i in range(0,35):
    if df.iat[i,1] == 1:
        channels.append(df.iat[i,3])
# get label for each sample
eu_label = pd.read_excel(r'EU_label.xlsx')



dirname_low = 'raw_fcs/flowrepo_covid_EU_00%d_flow_001'
dirname_high = 'raw_fcs/flowrepo_covid_EU_0%d_flow_001'
ext = ('.fcs')


full_data = []; count = 0
for i in range(2,49):
    if i <=9:
        dirname = dirname_low
    else:
        dirname = dirname_high
    isdir = os.path.isdir(dirname %i)
    if isdir == True:
        for files in os.listdir(dirname %i):
            if files.endswith(ext):
                p = pathlib.Path(files)
                s = FlowCal.io.FCSData(os.path.join(dirname %i, files))
                s = s[:, channels]
                s = np.asarray(s)
                added_s = np.empty_like(s, shape=(s.shape[0], 32))
                added_s[:, :-1] = s
                if eu_label.iat[count,1] == 'Healthy':
                    added_s[:, -1] = 0 # '0' means "healthy"
                else:
                    added_s[:, -1] = 1 # '1' means "sick"
                print(i, ", ", s.shape, ", label = ", added_s[0,31])
                full_data.extend(added_s)
                print("full_data: ", np.shape(full_data))
        count = count + 1



full_data = np.asarray(full_data)

np.random.shuffle(full_data)
print("final_full_data: ", np.shape(full_data))
# print(full_data[:,31])
# np.save('full_data_shuffle', full_data)
# np.savetxt('full_data_shuffle.txt', full_data, delimiter=',', fmt='%f')