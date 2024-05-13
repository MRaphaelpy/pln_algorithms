import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score

# Carregar os dados de um arquivo JSON
dados = pd.read_json('dados.json')

X = dados.drop('TARGET', axis=1)
y = dados['TARGET']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=42)
rus = RandomUnderSampler(random_state=42)

X_ros, y_ros = ros.fit_resample(X_treino, y_treino)
X_rus, y_rus = rus.fit_resample(X_treino, y_treino)

# Define the model
model = SVC()

# Fit and resample with ROS
model.fit(X_ros, y_ros)

# Cross-validation with ROS
pontuacao_ros = cross_val_score(model, X_ros, y_ros, cv=4)
print('Pontuações da validação cruzada com ROS: {}'.format(', '.join('{:.2f}'.format(p) for p in pontuacao_ros)))
print('Média da validação cruzada com ROS: {:.2f}'.format(pontuacao_ros.mean()))
print('Desvio padrão da validação cruzada com ROS: {:.2f}'.format(pontuacao_ros.std()))

# Fit and resample with RUS
model.fit(X_rus, y_rus)

# Cross-validation with RUS
try:
    pontuacao_rus = cross_val_score(model, X_rus, y_rus, cv=4)
    print('Pontuações da validação cruzada com RUS: {}'.format(', '.join('{:.2f}'.format(p) for p in pontuacao_rus)))
    print('Média da validação cruzada com RUS: {:.2f}'.format(pontuacao_rus.mean()))
    print('Desvio padrão da validação cruzada com RUS: {:.2f}'.format(pontuacao_rus.std()))
except ValueError:
    print('Não foi possível realizar a validação cruzada com RUS devido ao número insuficiente de amostras em uma ou mais classes.')