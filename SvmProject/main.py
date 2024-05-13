import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
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

modelos = {
    'SGDClassifier': SGDClassifier(),
    'SVC': SVC(),
   # 'LinearSVC': LinearSVC(),
    'svm.SVC': SVC()
}

for nome_modelo, modelo in modelos.items():
    modelo.fit(X_ros, y_ros)
    print('Modelo: {}\n\tPontuação com ROS: {:.2f}'.format(nome_modelo, modelo.score(X_teste, y_teste)))

    modelo.fit(X_rus, y_rus)
    print('\tPontuação com RUS: {:.2f}'.format(modelo.score(X_teste, y_teste)))

    pontuacao_ros = cross_val_score(modelo, X_ros, y_ros, cv=4)
    print('\tPontuações da validação cruzada com ROS: {}'.format(', '.join('{:.2f}'.format(p) for p in pontuacao_ros)))
    print('\tMédia da validação cruzada com ROS: {:.2f}'.format(pontuacao_ros.mean()))
    print('\tDesvio padrão da validação cruzada com ROS: {:.2f}'.format(pontuacao_ros.std()))

    try:
        pontuacao_rus = cross_val_score(modelo, X_rus, y_rus, cv=4)
        print('\tPontuações da validação cruzada com RUS: {}'.format(', '.join('{:.2f}'.format(p) for p in pontuacao_rus)))
        print('\tMédia da validação cruzada com RUS: {:.2f}'.format(pontuacao_rus.mean()))
        print('\tDesvio padrão da validação cruzada com RUS: {:.2f}'.format(pontuacao_rus.std()))
    except ValueError:
        print('\tNão foi possível realizar a validação cruzada para o modelo {} com RUS devido ao número insuficiente de amostras em uma ou mais classes.'.format(nome_modelo))
    print('\n' + '-'*50 + '\n')