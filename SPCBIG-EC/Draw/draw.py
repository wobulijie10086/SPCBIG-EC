# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/2 17:26
@Author  : Lee
@FileName: draw.py
@SoftWare: PyCharm
"""
# infinite
import csv
import matplotlib.pyplot as plt
# csv_file = csv.reader(open("all_dataset.CSV"))
# csv_file = csv.reader(open("dataset_CDAV.CSV"))
csv_file = csv.reader(open("dataset_integerUnderFlow.CSV"))
# csv_file = csv.reader(open("dataset_integer_big.CSV"))
epoch = []
accuracy = []
loss = []
accuracy_1 = []
loss_1 = []
for info in csv_file:
    if csv_file.line_num == 1:
        continue
    epoch.append(int(info[0]) + 1)
    accuracy.append(float(info[1]) * 1)
    loss.append(float(info[2]) * 1)
    accuracy_1.append(float(info[3]) * 1)
    loss_1.append(float(info[4]) * 1)

for info in csv_file:
    if csv_file.line_num == 1:
        continue

plt.plot(epoch, accuracy,lw=1, color='r', label='train acc', marker='.', markevery=2,
                 mew=1.5)
plt.plot(epoch, loss, lw=1, color='g', label='train loss', marker='.', markevery=2,
                 mew=1.5)
plt.plot(epoch, accuracy_1,lw=1, color='b', label='val-acc', marker='.', markevery=2,
                 mew=1.5)
plt.plot(epoch, loss_1, lw=1, color='darkorange', label='val-loss', marker='.', markevery=2,
                 mew=1.5)
plt.grid(True)
plt.xlabel("epoch", fontsize=12)
plt.ylabel("Accuracy-Loss", fontsize=12)
plt.legend(loc="center right")
plt.xlim(-0.1, 50)
plt.ylim(-0.01, 1.01)
plt.legend()
plt.show()