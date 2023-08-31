import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # para teste de normalidade.
from sklearn.tree import DecisionTreeClassifier # modelo de machine learn árvore de decisão.
from sklearn import tree #para gera a imagem da árvore de decisão.
from sklearn.metrics import accuracy_score, confusion_matrix #para medir a acuracia e a matrix de confusão do modelo.
from sklearn.metrics import precision_score #para medir a precisão do modelo.
from sklearn.metrics import recall_score #para medir a sensibilidade do modelo.
from sklearn.model_selection import train_test_split #divide o modelo em modelo de teste e treino.
from sklearn.ensemble import RandomForestClassifier # modelo de machine random forest.
from sklearn.ensemble import ExtraTreesClassifier # modelo de machine learn extra tree.
from sklearn.model_selection import GridSearchCV # funcão para escolher os melhores parametros do modelo.
from sklearn.model_selection import cross_val_score # modelo para fazer uma verificação cruzada nos modelos.
from sklearn.neighbors import KNeighborsClassifier # modelo de machine learn knn.


dados = pd.read_csv("D:/Estudos Python/bancos de dados/milknew.csv")

df = dados

descrição = df.describe()

correlacao = df.corr()

#%% classificação final 

sns.countplot(x = df['Grade'], order = ['low','medium','high'])

x = df['Grade'].value_counts()

#%% sabor do leite

taste_milk_bad = df.loc[df['Taste']==0, 'Grade']
taste_milk_good = df.loc[df['Taste']==1, 'Grade']

vc_taste_milk_bad = taste_milk_bad.value_counts()#.sort_index(), perguntar.
vc_taste_milk_good = taste_milk_good.value_counts()#.sort_index(), perguntar.

plt.figure()
plt.suptitle("Avalição do leite em relação ao gosto")
plt.subplot(1,2,1) 
plt.bar(x = vc_taste_milk_bad.index, height = vc_taste_milk_bad.values)
plt.ylim(0,500)
plt.xticks(['low','medium', 'high'], ['Baixo','Médio','Alto'])
plt.title('Leite de gosto ruim')

plt.subplot(1,2,2)
plt.bar(x=vc_taste_milk_good.index, height=vc_taste_milk_good.values, color='pink')
plt.ylim(0, 500)
plt.xticks(['low','medium', 'high'], ['Baixo','Médio','Alto'])
plt.yticks([])
plt.title('Leite de gosto bom')

#%% odor do leite

odor_milk_low = df.loc[df['Odor']==0, 'Grade']
odor_milk_high = df.loc[df['Odor']==1, 'Grade']

vc_odor_milk_low = odor_milk_low.value_counts()
vc_odor_milk_high = odor_milk_high.value_counts()

plt.figure()
plt.suptitle("Avalição do leite em relação ao odor")
plt.subplot(1,2,1) 
plt.bar(x = vc_odor_milk_low.index, height = vc_odor_milk_low.values)
plt.ylim(0,500)
plt.xticks(['medium','low', 'high'], ['Médio','Baixo','Alto'])
plt.title('Leite com baixo odor')

plt.subplot(1,2,2)
plt.bar(x=vc_odor_milk_high.index, height=vc_odor_milk_high.values, color='pink')
plt.ylim(0, 500)
plt.xticks(['medium','low', 'high'], ['Médio','Baixo','Alto'])
plt.yticks([])
plt.title('Leite com alto odor')

#%% Fat do leite

fat_milk_low = df.loc[df['Fat ']==0, 'Grade']
fat_milk_high = df.loc[df['Fat ']==1, 'Grade']

vc_fat_milk_low = fat_milk_low.value_counts()
vc_fat_milk_high = fat_milk_high.value_counts()

plt.figure()
plt.suptitle("Avalição do leite em relação ao nível de gordura")
plt.subplot(1,2,1) 
plt.bar(x = vc_fat_milk_low.index, height = vc_fat_milk_low.values)
plt.ylim(0,500)
plt.xticks(['medium','low', 'high'], ['Médio','Baixo','Alto'])
plt.title('Leite com pouca gordura')

plt.subplot(1,2,2)
plt.bar(x=vc_fat_milk_high.index, height=vc_fat_milk_high.values, color='pink')
plt.ylim(0, 500)
plt.xticks(['medium','low', 'high'], ['Médio','Baixo','Alto'])
plt.yticks([])
plt.title('Leite com muita gordura')

#%% turbidez do leite

turbidity_milk_low = df.loc[df['Turbidity']==0, 'Grade']
turbidity_milk_high = df.loc[df['Turbidity']==1, 'Grade']

vc_turbidity_milk_low = turbidity_milk_low.value_counts()
vc_turbidity_milk_high = turbidity_milk_high.value_counts()

plt.figure()
plt.suptitle("Avalição do leite em relação a turbidez")
plt.subplot(1,2,1) 
plt.bar(x = vc_turbidity_milk_low.index, height = vc_turbidity_milk_high.values)
plt.ylim(0,500)
plt.xticks(['medium','low', 'high'], ['Médio','Baixo','Alto'])
plt.title('Leite com baixa turbidez')

plt.subplot(1,2,2)
plt.bar(x=vc_turbidity_milk_high.index, height=vc_turbidity_milk_high.values, color='pink')
plt.ylim(0, 500)
plt.xticks(['medium','low', 'high'], ['Médio','Baixo','Alto'])
plt.yticks([])
plt.title('Leite com alto turbidez')

#%%temperatura

plt.figure()
leite_temperatura = df.groupby('Grade')['Temprature'].hist(alpha=0.50) # alpha determina a transparência do gráfico.
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Alta', 'baixo', 'Médio']) # adiciona uma legenda quando há mais de um gráfico no mesmo plot.
plt.title('Distribuição da Temperatura em relação as avaliações do leite')
plt.xlabel('Temperaturas')

plt.figure()
plt.scatter(df['Temprature'],df['Grade'])
plt.title('Distribuição da Temperatura em relação as avaliações do leite')

plt.figure()
sns.boxplot(x = df['Temprature'],y = df['Grade'], order = ['low','medium','high'])

#%% ph

plt.figure()
leite_temperatura = df.groupby('Grade')['pH'].hist(alpha=0.40) 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Alta', 'baixo', 'Médio']) 
plt.title('Distribuição do ph em relação as avaliações do leite')
plt.xlabel('PH do leite')

plt.figure()
plt.scatter(df['pH'],df['Grade'])
plt.title('Distribuição do ph em relação as avaliações do leite')

sns.boxplot(x = df['pH'],y = df['Grade'], order = ['low','medium','high'])

#%% cor

plt.figure()
leite_temperatura = df.groupby('Grade')['Colour'].hist(alpha=0.40) 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Alta', 'baixo', 'Médio']) 
plt.title('Distribuição do cor do leite em relação as avaliações do leite')
plt.xlabel('Cor do leite')

plt.figure()
plt.scatter(df['Colour'],df['Grade'])
plt.title('Distribuição do cor do leite em relação as avaliações do leite')

plt.figure()
sns.boxplot(x = df['Colour'],y = df['Grade'], order = ['low','medium','high'])

#%% teste de normalidade

plt.figure()
sns.displot(data=df['pH'], x=df['pH'], hue=df['Grade'], kind="kde")

stats.shapiro(df['Temprature'])
sns.displot(df['Temprature'], kind = 'kde')
sns.displot(data=df['Temprature'], x=df['Temprature'], hue=df['Grade'], kind="kde")


stats.shapiro(df['Taste'])
sns.displot(df['Taste'], kind = 'kde')
sns.displot(data=df['Taste'], x=df['Taste'], hue=df['Grade'], kind="kde",)

stats.shapiro(df['Odor'])
sns.displot(df['Odor'], kind = 'kde')
sns.displot(data=df['Odor'], x=df['Odor'], hue=df['Grade'], kind="kde")

stats.shapiro(df['Fat '])
sns.displot(df['Fat '], kind = 'kde')
sns.displot(data=df['Fat '], x=df['Fat '], hue=df['Grade'], kind="kde")

stats.shapiro(df['Turbidity'])
sns.displot(df['Turbidity'], kind = 'kde')
sns.displot(data=df['Turbidity'], x=df['Turbidity'], hue=df['Grade'], kind="kde")

stats.shapiro(df['Colour'])
sns.displot(df['Colour'], kind = 'kde')
sns.displot(data=df['Colour'], x=df['Colour'], hue=df['Grade'], kind="kde")

stats.shapiro(df['Grade'])
sns.displot(df['Grade'], kind = 'kde')

#%% machine learn

# em x estão todas as informações e em y estão as respostas para serem alcançadas
x = df.drop(columns=['Grade','Colour'])
y = df['Grade']

# melhores parametros DecisionTreeClassifier()
params = {'max_depth':[2,3,4,5,6,7,8],
          'criterion':['gini','entropy'],
          'class_weight':[None,'balanced'],
          'splitter': ['best', 'random']}

grid_search = GridSearchCV(estimator = DecisionTreeClassifier(),param_grid = params)

grid_search.fit(x,y)

melhores_parametros_DecisionTreeClassifier = grid_search.best_params_
melhor_resultado_DecisionTreeClassifier = grid_search.best_score_

# melhores parametros RandomForestClassifier

params_2 = {'criterion':['gini', 'entropy', 'log_loss'], 
            'max_features':['sqrt', 'log2', None]}

grid_search_2 = GridSearchCV(estimator = RandomForestClassifier(),param_grid = params_2)

grid_search_2.fit(x,y)

melhores_parametros_RandomForestClassifier = grid_search_2.best_params_
melhor_resultado_RandomForestClassifier = grid_search_2.best_score_

# melhores parametros ExtraTreesClassifier
params_3 = {'criterion':['gini', 'entropy', 'log_loss'],
            'max_features':['sqrt', 'log2', None]}

grid_search_3 = GridSearchCV(estimator = ExtraTreesClassifier(),param_grid = params_3)

grid_search_3.fit(x,y)

melhores_parametros_ExtraTreesClassifier = grid_search_3.best_params_
melhor_resultado_ExtraTreesClassifier = grid_search_3.best_score_

# melhores parametros knn

params_4 = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}

grid_search_4 = GridSearchCV(estimator = KNeighborsClassifier(),param_grid = params_4)

grid_search_4.fit(x,y)

melhores_parametros_KNeighborsClassifier = grid_search_4.best_params_
melhor_resultado_KNeighborsClassifier = grid_search_4.best_score_

#separando os dados em modelos de teste e treino

[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2, random_state = 0)

#árvore de decisão

modelo_1 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 8, splitter = 'random' , class_weight = 'balanced', random_state = 0) #forma da qual estou usando a árvore de decisão.
modelo_1.fit(x_train,y_train) # treinando o modelo para gerar resultados.
modelo_1.feature_importances_ # mostra a importância de cada variável.

tree.plot_tree(modelo_1) #para gerar a imagem da árvore decisão.

previsoes_1 = modelo_1.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_1 = accuracy_score(y_test, previsoes_1) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_1 = confusion_matrix (y_test,previsoes_1) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_1 = precision_score(y_test, previsoes_1, average = None) # teste da precisão do modelo.
recall_1 = recall_score(y_test, previsoes_1, average = None) # teste de sensibilidade do modelo.

previsoes_proba_1 = modelo_1.predict_proba(x_test)
#probalidade, brincar com if e else para definir qual é a melhor probalidade para cada peso de importancia.
#teste de ruido

scores_1 = cross_val_score(modelo_1, x, y, cv = 50, scoring = 'accuracy') # cross validation para melhor treinamento dos dados.

# modelo de decisão rando forest.
[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2 )

random_forest_df = RandomForestClassifier(n_estimators = 10, criterion = 'gini', max_features = 'sqrt',random_state = 0)
random_forest_df.fit(x_train, y_train)

previsoes_2 = random_forest_df.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_2 = accuracy_score(y_test, previsoes_2) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_2 = confusion_matrix (y_test,previsoes_2) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_2 = precision_score(y_test, previsoes_2, average = None) # teste da precisão do modelo.
recall_2 = recall_score(y_test, previsoes_2, average= None) # teste de sensibilidade do modelo.

scores_2 = cross_val_score(random_forest_df, x, y, cv = 50, scoring = 'accuracy') # cross validation para melhor treinamento dos dados.

# modelo de decisão extra tree.
[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2 )

extra_tree_df = ExtraTreesClassifier(criterion = 'gini', max_features = None)
extra_tree_df.fit(x_train, y_train)

previsoes_3  = extra_tree_df.predict(x_test) # jogando os dados de teste para tentar prever.
previsoes_3 = extra_tree_df.predict(x_test) # teste de acurácia para ver se o modelo foi eficiente.
acuracia_3 = accuracy_score(y_test, previsoes_3) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
Matriz_Confusao_3 = confusion_matrix (y_test,previsoes_3) # teste da precisão do modelo.
precision_3 = precision_score(y_test, previsoes_3, average = None) # teste da precisão do modelo.
recall_3 = recall_score(y_test, previsoes_3, average= None) # teste de sensibilidade do modelo.

scores_3 = cross_val_score(extra_tree_df, x, y, cv = 50, scoring = 'accuracy') # cross validation para melhor treinamento dos dados.

# modelo de decisão knn.
[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2 )

knn_modelo = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_modelo.fit(x_train,y_train)

previsoes_4  = knn_modelo.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_4 = accuracy_score(y_test, previsoes_4) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_4 = confusion_matrix (y_test,previsoes_4) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_4 = precision_score(y_test, previsoes_4, average = None) # teste da precisão do modelo.
recall_4 = recall_score(y_test, previsoes_4, average= None) # teste de sensibilidade do modelo.

scores_4 = cross_val_score(knn_modelo, x, y, cv = 50, scoring = 'accuracy') # cross validation para melhor treinamento dos dados.

# analise dos resultados

#%% função para prever dados futuros.

def novos_dados(a):
    resultado = random_forest_df.predict(a)
    return resultado

novos_dados = novos_dados(x_test) #testando a validade da função.

a1 = accuracy_score(y_test, novos_dados)

