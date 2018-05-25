import os
import pandas as pd

ctrl_path = "/home/pier/Machine_Learning/Clinical_Health_Parkinsons_Analytics/2016_img_data_analytics/data/control/"
control_files_list = [f for f in os.listdir(ctrl_path) if os.path.isfile(os.path.join(ctrl_path, f))]
park_path = "/home/pier/Machine_Learning/Clinical_Health_Parkinsons_Analytics/2016_img_data_analytics/data/parkinsons/"
parkinson_files_list = [f for f in os.listdir(park_path) if os.path.isfile(os.path.join(park_path, f))]

with open(ctrl_path + 'control_all.csv', 'w') as outfile:
    for fname in control_files_list:
        fname = ctrl_path + fname
        with open(fname) as infile:
            for line in infile:
                line = line.replace(';', ',')  # make it csv
                outfile.write(line)

with open(park_path + 'park_all.csv', 'w') as outfile:
    for fname in parkinson_files_list:
        fname = park_path + fname
        with open(fname) as infile:
            for line in infile:
                line = line.replace(';', ',')
                outfile.write(line)

df = pd.read_csv(ctrl_path + 'control_all.csv')
df.columns = ['X', 'Y', 'Z', 'Pressure', 'GripAngle', 'Timestamp', 'Test_ID']
df['PWP'] = 0
df.to_csv(ctrl_path + 'control_all_with_header.csv')

df = pd.read_csv(park_path + 'park_all.csv')
df.columns = ['X', 'Y', 'Z', 'Pressure', 'GripAngle', 'Timestamp', 'Test_ID']
df['PWP'] = 1
df.to_csv(park_path + 'park_all_with_header.csv')