import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# n_groups = 4
#
# means_men = (30, 60,20, 9.8)
#
# # std_men = (2, 3, 4, 1, 2)
#
# # means_women = (25, 32, 34, 20, 25)
# # std_women = (3, 5, 2, 3, 3)
#
#
# fig, ax = plt.subplots()
#
# index = np.arange(n_groups)
# bar_width = 0.4
#
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
#
# rects1 = ax.bar(index+bar_width/2, means_men, bar_width,
#                 alpha=opacity, color='g',
#                 # yerr=std_men,
#                 error_kw=error_config,
#                 )
#
# # rects2 = ax.bar(index + bar_width, means_women, bar_width,
# #                 alpha=opacity, color='r',
# #                 # yerr=std_women,
# #                 error_kw=error_config,
# #                 label='Women')
#
# ax.set_xlabel('Tools')
# ax.set_ylabel('Time(seconds)')
# ax.set_title('Detection Time Evaluation(seconds)')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('Oyente', 'Mythril','Securify', 'SPCBIG-EC',))
# ax.legend()
#
# fig.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

#Reentrancy
models = ('Eth2Vec','DR-GCN','TMP','CGE','AME', 'AFS', 'DeeSCV', 'Our model')
ACC = [85.50,81.47,84.48, 89.15,90.19, 93.07, 93.02,96.66]
TPR = [74.32,80.89,82.63, 87.62,89.69, 94.6, 83.46,98.04]
PRE = [86.60,72.36,74.06, 85.24,86.25, 90.0, 90.70,94.55]
F1 = [ 61.50,76.39,78.11, 86.41,87.94, 93.21, 86.87,96.74]
bar_width = 0.2
y_major_locator = MultipleLocator(5)
x = np.arange(len(models))
acc = x - 1.5 * bar_width
tpr = x - 0.5 * bar_width
pre = x + 0.5 * bar_width
f1 = x + 1.5 * bar_width
# 使用两次 bar 函数画出两组条形图
plt.bar(acc, height=ACC, width=bar_width, color='y', label='ACC', alpha=0.5, linewidth=10)
plt.bar(tpr, height=TPR, width=bar_width, color='g', label='TPR', alpha=0.5, linewidth=10)
plt.bar(pre, height=PRE, width=bar_width, color='gray', label='PRE', alpha=0.5, linewidth=10)
plt.bar(f1, height=F1, width=bar_width, color='orange', label='F1', alpha=0.5, linewidth=10)
for a, b in zip(acc, ACC):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(tpr, TPR):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(pre, PRE):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(f1, F1):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
plt.legend()  # 显示图例
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.set_xticklabels(labels=models, rotation=40)
plt.ylim(50, 110)
plt.xticks(x, labels=models, fontsize=9)
plt.title("Rentrancy",fontsize=15)
plt.ylabel('(%)')
plt.show()

#timestamp
models = ('DR-GCN','TMP','CGE','AME',  'DeeSCV', 'Our model')
ACC = [78.68, 83.45,89.02,86.52, 80.5,  91.11]
TPR = [78.91, 83.82,88.10,86.23, 74.86, 96.84]
PRE = [71.29, 75.05,87.41,82.07, 85.53, 86.93]
F1 = [74.91, 79.19, 87.75,84.10, 79.93, 91.62]
bar_width = 0.2
y_major_locator = MultipleLocator(5)
x = np.arange(len(models))
acc = x - 1.5 * bar_width
tpr = x - 0.5 * bar_width
pre = x + 0.5 * bar_width
f1 = x + 1.5 * bar_width
# 使用两次 bar 函数画出两组条形图
plt.bar(acc, height=ACC, width=bar_width, color='y', label='ACC', alpha=0.5, linewidth=10)
plt.bar(tpr, height=TPR, width=bar_width, color='g', label='TPR', alpha=0.5, linewidth=10)
plt.bar(pre, height=PRE, width=bar_width, color='gray', label='PRE', alpha=0.5, linewidth=10)
plt.bar(f1, height=F1, width=bar_width, color='orange', label='F1', alpha=0.5, linewidth=10)
for a, b in zip(acc, ACC):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(tpr, TPR):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(pre, PRE):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(f1, F1):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
plt.legend()  # 显示图例
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.set_xticklabels(labels=models, rotation=40)
plt.ylim(50, 110)
plt.xticks(x, labels=models, fontsize=9)
plt.title("Timestamp",fontsize=15)
plt.ylabel('(%)')
plt.show()

#infinite
models = ('DR-GCN','TMP','CGE','AME',  'Our model')
ACC = [68.34, 74.61, 83.21,80.32,  94.87]
TPR = [67.82, 74.32, 82.29,79.08,  93.16]
PRE = [64.89, 73.89, 81.97,78.69,  96.46]
F1 = [ 66.32, 74.10, 82.13,78.88,  95.00]
bar_width = 0.2
y_major_locator = MultipleLocator(5)
x = np.arange(len(models))
acc = x - 1.5 * bar_width
tpr = x - 0.5 * bar_width
pre = x + 0.5 * bar_width
f1 = x + 1.5 * bar_width
# 使用两次 bar 函数画出两组条形图
plt.bar(acc, height=ACC, width=bar_width, color='y', label='ACC', alpha=0.5, linewidth=10)
plt.bar(tpr, height=TPR, width=bar_width, color='g', label='TPR', alpha=0.5, linewidth=10)
plt.bar(pre, height=PRE, width=bar_width, color='gray', label='PRE', alpha=0.5, linewidth=10)
plt.bar(f1, height=F1, width=bar_width, color='orange', label='F1', alpha=0.5, linewidth=10)
for a, b in zip(acc, ACC):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(tpr, TPR):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(pre, PRE):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
for a, b in zip(f1, F1):
    plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=9, rotation=90)
plt.legend()  # 显示图例
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.set_xticklabels(labels=models, rotation=40)
plt.ylim(50, 110)
plt.xticks(x, labels=models, fontsize=9)
plt.title("Infinite loop",fontsize=15)
plt.ylabel('(%)')
plt.show()