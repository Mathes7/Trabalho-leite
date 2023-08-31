# correção dos gráficos

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dados = pd.read_csv("C:/Users/mathe/Downloads/milknew.csv")

df = dados

#%% tratamento para matplotlib

df['Grade'].replace('low', 0, inplace = True)
df['Grade'].replace('medium', 1, inplace = True)
df['Grade'].replace('high', 2, inplace = True)

#%% tratamento para seaborn

df.rename(columns={'Temprature':'Temperatura', 'Taste':'Sabor', 'Fat ': 'Gordura', 'Turbidity':'Turbidez', 'Colour':'Cor','Grade':'Avaliação'}, inplace = True)

df['Avaliação'].replace('low', 'Baixa', inplace = True)
df['Avaliação'].replace('medium', 'Média', inplace = True)
df['Avaliação'].replace('high', 'Alta', inplace = True)

#%% pH do leite

plt.figure()
leite_temperatura = df.groupby('Grade')['pH'].hist(alpha=0.40).sort_index() # sort_index() organiza os indices. 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Baixo', 'Médio', 'Alto']) 
plt.title('Distribuição do ph em relação as avaliações do leite')
plt.xlabel('PH do leite')

plt.figure()
plt.scatter(df['pH'],df['Grade'])
plt.xticks([0,1, 2], ['Baixo','Médio','Alto'])
plt.title('Distribuição do ph em relação as avaliações do leite')

plt.figure()
sns.boxplot(x = df['pH'],y = df['Avaliação'], order = ['Baixa','Média','Alta'])

plt.figure()
sns.displot(data=df['pH'], x=df['pH'], hue=df['Avaliação'], kind="kde")
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade do pH do leite')
#%% temperatura

plt.figure()
leite_temperatura = df.groupby('Avaliação')['Temperatura'].hist(alpha=0.50).sort_index() # alpha determina a transparência do gráfico.
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Baixo', 'Médio', 'Alta']) # adiciona uma legenda quando há mais de um gráfico no mesmo plot.
plt.title('Distribuição da Temperatura em relação as avaliações do leite')
plt.xlabel('Temperaturas')

plt.figure()
plt.scatter(df['Temperatura'],df['Avaliação'])
plt.title('Distribuição da Temperatura em relação as avaliações do leite')

plt.figure()
sns.boxplot(x = df['Temperatura'],y = df['Avaliação'], order = ['Baixa','Média','Alta'])

plt.figure()
sns.displot(data=df['Temperatura'], x=df['Temperatura'], hue=df['Avaliação'], kind="kde")
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade da temperatura do leite')
#%% sabor

taste_milk_bad = df.loc[df['Taste']==0, 'Grade']
taste_milk_good = df.loc[df['Taste']==1, 'Grade']

vc_taste_milk_bad = taste_milk_bad.value_counts().sort_index() # sort_index() organiza os indices.
vc_taste_milk_good = taste_milk_good.value_counts().sort_index() # sort_index() organiza os indices.

plt.figure()
plt.suptitle("Avalição do leite em relação ao gosto")
plt.subplot(1,2,1) 
plt.bar(x = vc_taste_milk_bad.index, height = vc_taste_milk_bad.values)
plt.ylim(0,500)
plt.xticks([0,1, 2], ['Baixo','Médio','Alto'])
plt.title('Leite de gosto ruim')

plt.subplot(1,2,2)
plt.bar(x=vc_taste_milk_good.index, height=vc_taste_milk_good.values, color='pink')
plt.ylim(0, 500)
plt.xticks([0,1,2], ['Baixo','Médio','Alto'])
plt.yticks([])
plt.title('Leite de gosto bom')

plt.figure()
sns.displot(data=df['Sabor'], x=df['Sabor'], hue=df['Avaliação'], kind="kde",)
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade do sabor do leite')

#%% odor

odor_milk_low = df.loc[df['Odor']==0, 'Grade']
odor_milk_high = df.loc[df['Odor']==1, 'Grade']

vc_odor_milk_low = odor_milk_low.value_counts().sort_index()
vc_odor_milk_high = odor_milk_high.value_counts().sort_index()

plt.figure()
plt.suptitle("Avalição do leite em relação ao odor")
plt.subplot(1,2,1) 
plt.bar(x = vc_odor_milk_low.index, height = vc_odor_milk_low.values)
plt.ylim(0,500)
plt.xticks([0,1, 2], ['Baixo', 'Médio','Alto'])
plt.title('Leite com baixo odor')

plt.subplot(1,2,2)
plt.bar(x=vc_odor_milk_high.index, height=vc_odor_milk_high.values, color='pink')
plt.ylim(0, 500)
plt.xticks([0,1, 2], ['Baixo', 'Médio','Alto'])
plt.yticks([])
plt.title('Leite com alto odor')

plt.figure()
sns.displot(data=df['Odor'], x=df['Odor'], hue=df['Avaliação'], kind="kde")
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade do odor do leite')
#%% gordura

fat_milk_low = df.loc[df['Fat ']==0, 'Grade']
fat_milk_high = df.loc[df['Fat ']==1, 'Grade']

vc_fat_milk_low = fat_milk_low.value_counts().sort_index()
vc_fat_milk_high = fat_milk_high.value_counts().sort_index()

plt.figure()
plt.suptitle("Avalição do leite em relação ao nível de gordura")
plt.subplot(1,2,1) 
plt.bar(x = vc_fat_milk_low.index, height = vc_fat_milk_low.values)
plt.ylim(0,500)
plt.xticks([0,1, 2], ['Baixo', 'Médio','Alto'])
plt.title('Leite com pouca gordura')

plt.subplot(1,2,2)
plt.bar(x=vc_fat_milk_high.index, height=vc_fat_milk_high.values, color='pink')
plt.ylim(0, 500)
plt.xticks([0,1, 2], ['Baixo', 'Médio','Alto'])
plt.yticks([])
plt.title('Leite com muita gordura')

plt.figure()
sns.displot(data=df['Gordura'], x=df['Gordura'], hue=df['Avaliação'], kind="kde")
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade da gordura do leite')
#%% turbidez

turbidity_milk_low = df.loc[df['Turbidity']==0, 'Grade']
turbidity_milk_high = df.loc[df['Turbidity']==1, 'Grade']

vc_turbidity_milk_low = turbidity_milk_low.value_counts().sort_index()
vc_turbidity_milk_high = turbidity_milk_high.value_counts().sort_index()

plt.figure()
plt.suptitle("Avalição do leite em relação a turbidez")
plt.subplot(1,2,1) 
plt.bar(x = vc_turbidity_milk_low.index, height = vc_turbidity_milk_high.values)
plt.ylim(0,500)
plt.xticks([0,1, 2], ['Baixo', 'Médio','Alto'])
plt.title('Leite com baixa turbidez')

plt.subplot(1,2,2)
plt.bar(x=vc_turbidity_milk_high.index, height=vc_turbidity_milk_high.values, color='pink')
plt.ylim(0, 500)
plt.xticks([0,1, 2], ['Baixo', 'Médio','Alto'])
plt.yticks([])
plt.title('Leite com alto turbidez')

plt.figure()
sns.displot(data=df['Turbidez'], x=df['Turbidez'], hue=df['Avaliação'], kind="kde")
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade da turbidez do leite')
#%% cor

plt.figure()
leite_temperatura = df.groupby('Grade')['Colour'].hist(alpha=0.40) 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Alta', 'baixo', 'Médio']) 
plt.title('Distribuição do cor do leite em relação as avaliações do leite')
plt.xlabel('Cor do leite')

plt.figure()
sns.boxplot(x = df['Cor'],y = df['Avaliação'], order = ['Baixa','Média','Alta'])

plt.figure()
plt.scatter(df['Cor'],df['Avaliação'])
plt.title('Distribuição do cor do leite em relação as avaliações do leite')

plt.figure()
sns.displot(data=df['Cor'], x=df['Cor'], hue=df['Avaliação'], kind="kde")
plt.ylabel('Densidade')
plt.title('Distribuição de normalidade da cor do leite')

#%% grade

sns.countplot(x = df['Avaliação'], order = ['Baixa','Média','Alta'])
plt.title('Amostras do leite')
plt.ylabel('Quantidade')