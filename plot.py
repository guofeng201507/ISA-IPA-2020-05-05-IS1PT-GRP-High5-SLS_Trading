import pandas as pd
import matplotlib.pyplot as plt



df_train = pd.read_csv('./stockdata/train/SPY_training.csv')

fig_train = df_train.plot(x ='Date', y='Close', kind = 'line').get_figure()

fig_train.autofmt_xdate()

fig_train.savefig("./stockdata/train/SPY_training.png")

plt.show()

df_test = pd.read_csv('./stockdata/test/SPY_testing.csv')

fig_test = df_test.plot(x ='Date', y='Close', kind = 'line').get_figure()
fig_test.autofmt_xdate()
fig_test.savefig("./stockdata/train/SPY_testing.png")

plt.show()