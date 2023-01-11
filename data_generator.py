'''
This code is to create the full data set to a *.npy file with the size of [5052720, 32] (5052720 samples)
- In one sample, the format is [input, label] in which the input includes 31 features and label has 1 value with '0' or '1',
it means, the size of one sample is [1,32]
- To run this code, just make sure you put the following files in the same folder with "data_generator.py"
1. raw_fcs
2. EU_label.xlsx
3. EU_marker_channel_mapping.xlsx

--> after running this code, it will automatically generate a data set file "full_data_shuffle.npy"
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
for i in range(2,49): # this command is to load all folder including *.fcs file
    if i <=9: # because the way to index folder name is not gradually increased, this condition is used to
        # assign suitable path names
        dirname = dirname_low
    else:
        dirname = dirname_high
    isdir = os.path.isdir(dirname %i)
    if isdir == True:
        for files in os.listdir(dirname %i):
            if files.endswith(ext): # to check if existing a *.fcs file inside the current folder
                p = pathlib.Path(files)
                s = FlowCal.io.FCSData(os.path.join(dirname %i, files))
                s = s[:, channels]
                s = np.asarray(s)

                added_s = np.empty_like(s, shape=(s.shape[0], 32)) # this part is to add the last column for label
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

np.random.shuffle(full_data) # to shuffle data set
print("final_full_data: ", np.shape(full_data))
# print(full_data[:,31])
np.save('full_data_shuffle', full_data) # to save the full data set into *.npy file
# np.savetxt('full_data_shuffle.txt', full_data, delimiter=',', fmt='%f') # if you want to save to *.txt file for checking purpose