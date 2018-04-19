# Each plot is 600 pixels wide, one pixel is one second, so the width represents 10 minutes.
# The pixel value is the log of the click count for that second, so one hour is 6 rows
# of pixels, and one day is 144 rows.
# 
# More details here:
# https://www.kaggle.com/jtrotman/eda-talkingdata-temporal-click-count-plots
# 

import numpy as np
import pandas as pd
import os, sys, time
from imageio import imwrite

# redefine this to run locally and generate 30-50 plots per column
PLOT_COUNT = 5

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16'
        }

# read a subset, runs out of memory otherwise (os seems least interesting)
fields = [ 'ip', 'app', 'device', 'channel' ]  # , 'os' ]
to_read = fields + [ 'click_time' ]

train_df  = pd.read_csv('../input/train.csv', usecols=to_read, parse_dates=['click_time'], dtype=dtypes) # , nrows=720000)
test_df   = pd.read_csv('../input/test.csv', usecols=to_read, parse_dates=['click_time'], dtype=dtypes)

print('Loaded', train_df.shape, test_df.shape)

def datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.int32)


# e.g. train_df.loc[train_df.ip==234]
def generate_plot(df, name):
    w = 600
    n = df.sec.max()+1
    l = int(np.ceil(n/float(w))*w)
    c = np.zeros(l, dtype=np.float32)
#   np.add.at(c, df.sec.values, df.is_attributed.values) # use this to plot target value instead
    np.add.at(c, df.sec.values, 1)
    print(f'\t {name} total clicks {c.sum():.0f} \t max clicks {c.max():.0f} \t mean click rate {c.mean():.02f} ')
    c = np.log1p(c)
    c /= c.max()
    imwrite(f'{name}.png', c.reshape((-1,w)))


train_df['sec'] = datetime_to_deltas(train_df.click_time)
test_df['sec'] = datetime_to_deltas(test_df.click_time)
print('Added seconds')

train_df.drop('click_time', axis=1, inplace=True)
test_df.drop('click_time', axis=1, inplace=True)
print('Dropped click_time')

for name, df in [ ('train', train_df), ('test', test_df) ]:
    for col in fields:
        print(f'Generating plots for top {col}s in {name}')
        counts = df[col].value_counts()
        for i, (v, c) in enumerate(counts.head(PLOT_COUNT).iteritems(), 1):
            # save with name e.g. train_channel_01_280.png
            # - 1st number is rank (1 = most common)
            # - 2nd number is field value itself
            generate_plot(df.loc[df[col]==v], f'{name}_{col}_{i:02d}_{v}')
