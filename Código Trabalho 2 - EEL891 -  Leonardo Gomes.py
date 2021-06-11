
import pandas as pd
import math
from sklearn.preprocessing import LabelBinarizer

from pandas_profiling import ProfileReport

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pycaret.regression import *



data = pd.read_csv('conjunto_de_treinamento.csv')
test = pd.read_csv('conjunto_de_teste.csv')

data.head(10)

data = data.replace(np.nan,0)
data = data.replace('',0)

test = test.replace(np.nan,0)
test = test.replace('',0)

binarizer = LabelBinarizer()
for indice in ['tipo_vendedor']:
    data[indice] = binarizer.fit_transform(data[indice])

binarizer = LabelBinarizer()
for indice in ['tipo_vendedor']:
    test[indice] = binarizer.fit_transform(test[indice])



data.head(10)


sns.set(font_scale=10)
f = plt.figure(figsize=(12, 12))

sns.set_theme(style="white")

heatmap = sns.heatmap(data.corr(),vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=1, linecolor='black',square=True)

heatmap.set_title('Correlation', fontdict={'fontsize':20}, pad=18)

size = heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 18)
size = heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 18)

id_solicitante_test = test["Id"].to_frame()

data = data.drop(['Id'],axis=1)
test = test.drop(['Id'],axis=1)

rgs = setup(data = data,  target = 'preco', silent = True, session_id = 123)

model_compare = compare_models(n_select = 5, sort = 'RMSLE')


model = create_model('huber')

tuned_model = tune_model(model, optimize = 'RMSLE')

evaluate_model(tuned_model)

predict = predict_model(tuned_model)

predictions = predict_model(tuned_model, data=test)
predictions.head(5)

result = {'Id': id_solicitante_test['Id'], 'preco': predictions['Label'].tolist()}
result_frame = pd.DataFrame(data=result)
result_frame.to_csv (r'answer.csv', index = False, header=True)




