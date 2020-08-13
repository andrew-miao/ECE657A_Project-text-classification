import matplotlib.pyplot as plt
import Exp02
import numpy as np


cnn = []
rcnn = []
lstm_attn = []
lstm_attn_gru = []
datasets = ['AGNews', 'Dbpedia', 'Amazon', 'Yelp']
for dataset in datasets:
    test_list = Exp02.experiment(dataset)
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

ax.set_ylabel('Macro-F1 Score')
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
