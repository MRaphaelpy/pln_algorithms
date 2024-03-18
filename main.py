import json
from collections import Counter
import spacy
from spacy.tokenizer import Tokenizer
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Carregar o modelo do SpaCy para o português
nlp = spacy.load('pt_core_news_sm')
tokenizer = Tokenizer(nlp.vocab)


# Carregar a lista de stop words em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Adicionar stop words personalizadas
stop_words_custom = {'nao', 'ter', '  ', "manha", '...', 'bem'}  # Adicione suas palavras aqui

# Combinar as stop words padrão com as personalizadas
stop_words.update(stop_words_custom)

def lemmatizar_spacy(texto):
    doc = nlp(texto)
    return ' '.join([token.lemma_ for token in doc])

def calculate_bow(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    return dict(word_counts)

# Nome do arquivo JSON de entrada
nome_arquivo_entrada = 'Dados_sem_acentos.json'

# Abrir o arquivo JSON de entrada
with open(nome_arquivo_entrada, 'r', encoding='utf-8') as arquivo_entrada:
    dados = json.load(arquivo_entrada)

# Lematizar os textos do JSON usando SpaCy
lemmatizados = [lemmatizar_spacy(dado['Comentario']) for dado in dados]

# Tokenizar os textos lematizados e remover pontuação e stop words
tokens = []
for texto in lemmatizados:
    doc = tokenizer(texto.lower())
    tokens.extend([token.text.lower() for token in doc if token.text.lower() not in stop_words and token.text.lower() not in string.punctuation])

# Contagem dos tokens mais frequentes
contagem_tokens = Counter(tokens)

# Obter os 10 tokens mais frequentes
top_10_tokens = contagem_tokens.most_common(10)

print('Top 10 tokens mais frequentes:')
for token, frequencia in top_10_tokens:
    print(f'Token: {token}, Frequência: {frequencia}')
print("\n")

# Calcular Bag of Words (BoW)
bow = calculate_bow(lemmatizados)

# Calcular TF-IDF usando TfidfVectorizer do scikit-learn
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(lemmatizados)
feature_names = list(vectorizer.get_feature_names_out())  # Converter para lista

# Mostrar cada palavra com seu BoW e TF-IDF
for i, token in enumerate(top_10_tokens):
    word_index = feature_names.index(token[0])
    bow_value = bow.get(token[0], 0)
    tfidf_value = X[0, word_index]
    print(f'Token: {token[0]:<10} | BoW: {bow_value:<6} | TF-IDF: {tfidf_value:<10}')

# Análise de Sentimento
notas = [dado['Nota'] for dado in dados]
media_notas = np.mean(notas)
print(f'\nMédia das notas: {media_notas}')

# Correlação entre Notas e Palavras-Chave
df = pd.DataFrame({
    'Nota': notas,
    'Comentario': lemmatizados
})

# Remover a coluna 'Comentario' antes de calcular a correlação
df = df.drop(columns=['Comentario'])

correlacao = df.corr()
print(f'\nCorrelação entre Notas e Palavras-Chave:\n{correlacao}')


# Carregar o modelo do SpaCy para o português
nlp = spacy.load('pt_core_news_sm')
tokenizer = Tokenizer(nlp.vocab)

# Carregar a lista de stop words em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Adicionar stop words personalizadas
stop_words_custom = {'nao', 'ter', '  ', "manha", '...', 'bem'}  # Adicione suas palavras aqui

# Combinar as stop words padrão com as personalizadas
stop_words.update(stop_words_custom)

def lemmatizar_spacy(texto):
    doc = nlp(texto)
    return ' '.join([token.lemma_ for token in doc])

def calculate_bow(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    return dict(word_counts)

# Nome do arquivo JSON de entrada
nome_arquivo_entrada = 'Dados_sem_acentos.json'

# Abrir o arquivo JSON de entrada
with open(nome_arquivo_entrada, 'r', encoding='utf-8') as arquivo_entrada:
    dados = json.load(arquivo_entrada)

# Lematizar os textos do JSON usando SpaCy
lemmatizados = [lemmatizar_spacy(dado['Comentario']) for dado in dados]

# Tokenizar os textos lematizados e remover pontuação e stop words
tokens = []
for texto in lemmatizados:
    doc = tokenizer(texto.lower())
    tokens.extend([token.text.lower() for token in doc if token.text.lower() not in stop_words and token.text.lower() not in string.punctuation])

# Contagem dos tokens mais frequentes
contagem_tokens = Counter(tokens)

# Obter os 10 tokens mais frequentes
top_10_tokens = contagem_tokens.most_common(10)

print('Top 10 tokens mais frequentes:')
for token, frequencia in top_10_tokens:
    print(f'Token: {token}, Frequência: {frequencia}')
print("\n")

# Calcular Bag of Words (BoW)
bow = calculate_bow(lemmatizados)

# Calcular TF-IDF usando TfidfVectorizer do scikit-learn
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(lemmatizados)
feature_names = list(vectorizer.get_feature_names_out())  # Converter para lista

# Mostrar cada palavra com seu BoW e TF-IDF
for i, token in enumerate(top_10_tokens):
    word_index = feature_names.index(token[0])
    bow_value = bow.get(token[0], 0)
    tfidf_value = X[0, word_index]
    print(f'Token: {token[0]:<10} | BoW: {bow_value:<6} | TF-IDF: {tfidf_value:<10}')

# Análise de Sentimento
notas = [dado['Nota'] for dado in dados]
media_notas = np.mean(notas)
print(f'\nMédia das notas: {media_notas}')

# Correlação entre Notas e Palavras-Chave
df = pd.DataFrame({
    'Nota': notas,
    'Comentario': lemmatizados
})

# Remover a coluna 'Comentario' antes de calcular a correlação
df = df.drop(columns=['Comentario'])

correlacao = df.corr()
print(f'\nCorrelação entre Notas e Palavras-Chave:\n{correlacao}')