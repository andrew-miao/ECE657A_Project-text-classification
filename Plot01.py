import matplotlib.pyplot as plt
import Exp01
import sys
import numpy as np


if len(sys.argv) != 2:
	sys.exit('Use: python Plot01.py [loss or acc]')

plot_obj = sys.argv[1]
if plot_obj != 'loss' and plot_obj != 'acc':
    sys.exit('Use: python Plot01.py [loss or acc]')
cnn = []
rcnn = []
lstm_attn = []
lstm_attn_gru = []
datasets = ['AGNews', 'Dbpedia', 'Amazon', 'Yelp']
for dataset in datasets:
    if plot_obj == 'loss':
        test_list, _ = Exp01.experiment(dataset)
    else:
        _, test_list = Exp01.experiment(dataset)
    cnn.append(test_list[0])
    rcnn.append(test_list[1])
    lstm_attn.append(test_list[2])
    lstm_attn_gru.append(test_list[3])

x = np.arange(4)
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.2, cnn, width, label='CNN')
rects2 = ax.bar(x, rcnn, width, label='RCNN')
rects3 = ax.bar(x + 0.2, lstm_attn, width, label='LSTM+Attn')
rects4 = ax.bar(x + 0.4, lstm_attn_gru, width, label='LSTM+Attn+GRU')

if plot_obj == 'loss':
    ax.set_title('Loss value for 4 models in 4 testing datasets')
    ax.set_ylabel('Loss Value')
else:
    ax.set_title('Accuracy for 4 models in 4 testing datasets')
    ax.set_ylabel('Accuracy Rate')

ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
fig.tight_layout()
ax.legend(loc='best')
'''
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
'''
plt.show()
