from pyecharts import Bar3D
import pandas

# data = pandas.DataFrame({'evaluate':['Acc','F1','Recall','FPR',
#                                      'Acc','F1','Recall','FPR',
#                                      'Acc','F1','Recall','FPR',
#                                      'Acc','F1','Recall','FPR',
#                                      'Acc','F1','Recall','FPR', ],
#                          'model':['lstm','lstm','lstm','lstm',
#                                  'blstm','blstm','blstm','blstm',
#                                  'blstm-att','blstm-att','blstm-att','blstm-att',
#                                  'bigru-att','bigru-att','bigru-att','bigru-att',
#                                   'spcnn_bigru-att', 'spcnn_bigru-att', 'spcnn_bigru-att', 'spcnn_bigru-att',],
#                          'value':[0.8173, 0.8006,0.7641,0.1196,
#                                   0.8538,0.8555,0.8657,0.1142,
#                                   0.8847,0.8826,0.8848,0.0857,
#                                   0.8857,0.8811,0.8476,0.0761,
#                                   0.9666, 0.9668,0.9714, 0.0381 ]
#
# }
# )

data = pandas.DataFrame({'evaluate':[
                                     'Acc','F1','Recall','Precision',
                                     'Acc','F1','Recall','Precision',
                                     'Acc','F1','Recall','Precision',
                                     'Acc','F1','Recall','Precision',
                                     # 'Acc','F1','Recall','Precision',
                                     # 'Acc','F1','Recall','Precision',
                                     'Acc','F1','Recall','Precision',
                                     'Acc','F1','Recall','Precision',
                                     ],
                         'model':[
                                 # 'lstm','lstm','lstm','lstm',
                                 # 'blstm','blstm','blstm','blstm',
                                  'DR-GCN','DR-GCN','DR-GCN','DR-GCN',
                                  'TMP','TMP','TMP','TMP',
                                  'blstm-att','blstm-att','blstm-att','blstm-att',
                                  'bigru-att','bigru-att','bigru-att','bigru-att',

                                  'CGE','CGE','CGE','CGE',
                                  'spcnn_bigru-att', 'spcnn_bigru-att', 'spcnn_bigru-att', 'spcnn_bigru-att',],
                         'value':[
                                  # 0.8173,0.8006,0.7641,0.8816,
                                  # 0.8538,0.8555,0.8657,0.8523,
                                  0.8147, 0.7639, 0.8089, 0.7236,
                                  0.8448, 0.7811, 0.8263, 0.7406,
                                  0.8847,0.8826,0.8848,0.8850,
                                  0.8857,0.8811,0.8476,0.9175,

                                  0.8915,0.8641,0.8762,0.8524,
                                  0.9666,0.9668,0.9714,0.9622,]

}
)

print(data)
x_name = list(set(data.iloc[:, 0]))
y_name = list(set(data.iloc[:, 1]))
print(x_name)
print(y_name)
data_xyz=[]
range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
               '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
for i in range(len(data)):

     x=x_name.index(data.iloc[i,0])

     y=y_name.index(data.iloc[i,1])

     z=data.iloc[i,2]

     data_xyz.append([x,y,z])
print(data_xyz)

bar3d=Bar3D("重入漏洞模型性能比较",title_pos="center",width=1200,height=800)
bar3d.add('',x_name,y_name,data_xyz,is_label_show=True,is_visualmap=True, visual_range_color='#ffffbf',grid3d_shading="lambert",visual_range=[0, 500],grid3d_width=150, grid3d_depth=50)
bar3d.render("reentrancy.html")