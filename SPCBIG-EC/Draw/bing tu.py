# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

#UCESC
labels='Callstack Depth Attack:1378','Integer Overflow:1640','Integer Underflow:1988','Reentrancy:2000','Timestamp Dependency:1671','Infinite Loop:1371','No Vulnerabilities:10048'

sizes=1378,1640,1988,2000,1671,1371,10048

colors='lightgreen','lightskyblue','lightcoral','darkorange','yellow','pink','lightgray'


explode=0.1,0.1,0.1,0.1,0.1,0.1,0.1

plt.pie(sizes,explode=explode,labels=labels,

        colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)

plt.axis('equal')
plt.title("UCESC-dataset",fontsize=15)
plt.show()

# #callstack
# labels='Callstack Depth Attack:1378','No Vulnerabilities:1378'
#
# sizes=1378,1378
#
# colors='lightgreen','lightgray'
#
#
# explode=0.1,0.1
#
# plt.pie(sizes,explode=explode,labels=labels,
#
#         colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
#
# plt.axis('equal')
# plt.title("Callstack Depth Attack",fontsize=15)
# plt.show()



# #Integer Overflow:1640
# labels='Integer Overflow:1640','No Vulnerabilities:1640'
#
# sizes=1640,1640
#
# colors='lightskyblue','lightgray'
#
#
# explode=0.1,0.1
#
# plt.pie(sizes,explode=explode,labels=labels,
#
#         colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
#
# plt.axis('equal')
# plt.title("Integer Overflow",fontsize=15)
# plt.show()
#
# #Integer Underflow:1988
# labels='Integer Underflow:1988','No Vulnerabilities:1988'
#
# sizes=1988,1988
#
# colors='lightcoral','lightgray'
#
#
# explode=0.1,0.1
#
# plt.pie(sizes,explode=explode,labels=labels,
#
#         colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
#
# plt.axis('equal')
# plt.title("Integer Underflow",fontsize=15)
# plt.show()
#
# #Reentrancy:2000
# labels='Reentrancy:2000','No Vulnerabilities:2000'
#
# sizes=2000,2000
#
# colors='darkorange','lightgray'
#
#
# explode=0.1,0.1
#
# plt.pie(sizes,explode=explode,labels=labels,
#
#         colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
#
# plt.axis('equal')
# plt.title("Reentrancy",fontsize=15)
# plt.show()
#
# #Timestamp Dependency
# labels='Timestamp Dependency:1671','No Vulnerabilities:1671'
#
# sizes=1671,1671
#
# colors='yellow','lightgray'
#
#
# explode=0.1,0.1
#
# plt.pie(sizes,explode=explode,labels=labels,
#
#         colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
#
# plt.axis('equal')
# plt.title("Timestamp Dependency",fontsize=15)
# plt.show()
#
# #Infinite Loop:1371
# labels='Infinite Loop:1371','No Vulnerabilities:1671'
#
# sizes=1371,1371
#
# colors='pink','lightgray'
#
#
# explode=0.1,0.1
#
# plt.pie(sizes,explode=explode,labels=labels,
#
#         colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
#
# plt.axis('equal')
# plt.title("Infinite Loop",fontsize=15)
# plt.show()