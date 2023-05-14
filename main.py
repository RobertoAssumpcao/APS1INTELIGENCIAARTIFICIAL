# Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Importar dados
data = pd.read_csv('/content/heart - heart.csv')

# Transformar variáveis categóricas em numéricas
data_num = pd.get_dummies(data, columns=['Sexo', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Visualizar dados numéricos
print(data_num.head())

# Dividir dados em treino, teste e avaliação
X_train, X_test, y_train, y_test = train_test_split(data_num.iloc[:,:-1], data_num.iloc[:,-1], test_size=0.3, random_state=42)
X_testResult, X_eval, y_testResult, y_eval = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Construir árvore de decisão
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Avaliar desempenho no conjunto de avaliação
y_pred = tree.predict(X_eval)
accuracy = accuracy_score(y_eval, y_pred)
print('Porcentagem de acertos na avaliação:', accuracy)

# Salvar imagem da árvore de decisão
plot_tree(tree, feature_names=data_num.columns[:-1], class_names=['0', '1'], filled=True, rounded=True)
plt.savefig('tree.png')
