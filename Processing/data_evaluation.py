# %%

import pandas as pd  # for data processing CSV data
import tensorflow as tf
import numpy as np  # for linear algebra
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 1, 1

RANDOM_SEED = 42

columns = ['user', 'activity', 'x-axis', 'y-axis', 'z-axis']
# dataframe
df = pd.read_csv('data/reduced.txt',
                 header=None, names=columns)
df.head()

# exploring the data
df['activity'].value_counts().plot(
    kind='bar', title='Training examples by activity type')
df['user'].value_counts().plot(
    kind='bar', title='Training examples by user')

# custom function to plot


def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


# graph showing activity using function
plot_activity("Sitting", df)
plot_activity("Standing", df)
plot_activity("Walking", df)
plot_activity("Jogging", df)
