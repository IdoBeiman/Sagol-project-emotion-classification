import pandas as pd
import numpy as np
import sys

file_path = sys.argv[1]
output_path = file_path.split('.')[0] + '.txt'
df = pd.read_csv(file_path)
df.drop(df.columns[[0, 1]], axis=1, inplace=True)
values = [df.columns.values.tolist()] + df.values.tolist()
np.savetxt(output_path, values, fmt='%s', newline='')