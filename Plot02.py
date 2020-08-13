import matplotlib.pyplot as plt
import Exp01
import sys
import numpy as np


if len(sys.argv) != 2:
	sys.exit('Use: python Plot01.py [dataset name]')

dataset_name = sys.argv[1]
if dataset_name != 'AGNews' and dataset_name != 'Dbpedia':
    sys.exit('Now we only support AGNews and Dbpedia datasets')

cnn = []
rcnn = []
lstm_attn = []
lstm_attn_gru = []
ratio_labels = ['20%', '40%', '60%', '80%', '100%']
ratio = [0.2, 0.4, 0.6, 0.8, 1.0]

x = np.arange(5)
for data_size in ratio:
    _, test_list = Exp01.experiment('AGNews', data_size=data_size)
    cnn.append(test_list[0])
    rcnn.append(test_list[1])
    lstm_attn.append(test_list[2])
    lstm_attn_gru.append(test_list[3])

plt.plot(x, cnn, label='CNN')
plt.plot(x, rcnn, label='RCNN')
plt.plot(x, lstm_attn, label='LSTM+Attn')
plt.plot(x, lstm_attn_gru, label='LSTM+Attn+GRU')

plt.title('')
ax = plt.gca()
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(ratio_labels)
xlabel = 'The data proportions of ' + dataset_name + '.'
plt.xlabel(xlabel)
plt.ylabel('Accuracy rate (%)')
ax.legend(loc='best')
plt.show()