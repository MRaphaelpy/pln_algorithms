import json
from collections import Counter
import spacy
from spacy.tokenizer import Tokenizer
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import re
import pandas as pd
from unidecode import unidecode

# carregar o modelo do SpaCy para o portugues
nlp = spacy.load('pt_core_news_sm')
tokenizer = Tokenizer(nlp.vocab)

# carregar a lista de stop words em portugues
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# cdicionar stop words personalizadas
stop_words_custom = {'nao', 'ter', '  ', "manha", '...', 'bem', "!"}  # Adicione suas palavras aqui

# combinar as stop words padrao com as personalizadas
stop_words.update(stop_words_custom)

# funcao para lematizar o texto usando SpaCy
def lemmatizar_spacy(texto):
    doc = nlp(texto.lower())
    lemmas = ' '.join([token.lemma_ for token in doc])
    lemmas = re.sub(r'[^\w\-]+|\d+', ' ', lemmas)
    return lemmas.lower()

# Funcao para calcular  o bow <- arco ?
def calculate_bow(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update([unidecode(word) for word in text])
    return dict(word_counts)

# Funcao para calcular TF-IDF
def calculate_tfidf(bow, texts):
    tfidf = {}
    total_texts = len(texts)
    for word, count in bow.items():
        tf = count / total_texts
        idf = np.log(total_texts / (1 + count))
        tfidf[word] = tf * idf
    return tfidf

# funcao para buscar um termo nos documentos
def search_term(term, tfidf):
    return sorted(tfidf.items(), key=lambda x: x[1], reverse=True) if term in tfidf else []

# funcao para criar um vetor BoW
def create_bow_vector(dictionary, comment_list):
    return [1 if word in comment_list else 0 for word in dictionary]

# Funcao para criar um vetor TF-IDF
def create_tfidf_vector(dictionary, comment_list, tfidf):
    return [tfidf[word] if word in comment_list else 0 for word in dictionary]

# Funcao para encontrar uma palavra em um dicionario
def find_word(dictionary, index):
    return dictionary[index]

def substituir_abreviacoes(tokens, abreviacoes):
    return [abreviacoes.get(token, token) for token in tokens]

abreviacoes = {'vc': 'você', 'tb': 'tambem', 'q': 'que', 'n': 'nao', 'td':'tudo', 'wi-fi':'wifi', 'hj':'hoje', 'vdd':'verdade', 'mto':'muito', 'pq':'porque'}  # Coloque as abreviacoes aqui



# Nome do arquivo JSON de entrada
nomeArquivoJson = 'Dados_sem_acentos.json'

# Abrir o arquivo JSON de entrada
with open(nomeArquivoJson, 'r', encoding='utf-8') as arquivo_entrada:
    dados = json.load(arquivo_entrada)

# Lematizar os textos do JSON usando SpaCy
lemmatizados = [lemmatizar_spacy(dado['Comentario']).split() for dado in dados]
lemmatizados = [substituir_abreviacoes(texto, abreviacoes) for texto in lemmatizados]

# Tokenizar os textos lematizados e remover pontuacao e stop words
tokens = []
for texto in lemmatizados:
    doc = tokenizer(' '.join(texto).lower())
    tokens.extend([token.text.lower() for token in doc if token.text.lower() not in stop_words and token.text.lower() not in string.punctuation])

# Contagem dos tokens mais frequentes
contagem_tokens = Counter(tokens)

# Obter os 10 tokens mais frequentes
top_10_tokens = contagem_tokens.most_common(10)

print('Top 10 tokens mais frequentes:')
for token, frequencia in top_10_tokens:
    print(f'Token: {token}, Frequencia: {frequencia}')
print("\n")

# Calcular Bag of Words (BoW)
bow = calculate_bow(lemmatizados)

# Calcular TF-IDF
tfidf = calculate_tfidf(bow, lemmatizados)

# Mostrar cada palavra com seu BoW e TF-IDF
for i, token in enumerate(top_10_tokens):
    bow_value = bow.get(token[0], 0)
    tfidf_value = tfidf.get(token[0], 0)
    print(f'Token: {token[0]:<10} | BoW: {bow_value:<6} | TF-IDF: {tfidf_value:<10}')

# Analise de Sentimento
notas = [dado['Nota'] for dado in dados]
media_notas = np.mean(notas)
print(f'\nMedia das notas: {media_notas}')

# Correlacao entre Notas e Palavras-Chave
df = pd.DataFrame({
    'Nota': notas,
    'Comentario': lemmatizados
})

# Remover a coluna 'Comentario' antes de calcular a correlacao
df = df.drop(columns=['Comentario'])

correlacao = df.corr()
print(f'\nCorrelacao entre Notas e Palavras-Chave:\n{correlacao}')

# Buscar termo
termo = 'bom'  # Altere para o termo desejado
resultados = search_term(termo, tfidf)
print(f'\nDocumentos mais relevantes para o termo "{termo}":')
for i, (doc, score) in enumerate(resultados[:10]):
    print(f'{i+1}. Documento: {doc}, Score: {score}')
    
dictionary = list(bow.keys())

comment_list = ['verdade']  # coloque as palavras aqui


# Calcular vetor BoW
bow_vector = create_bow_vector(dictionary, comment_list)
print(f'Vetor BoW: {bow_vector}')

print("\n")

# Calcular vetor TF-IDF
tfidf_vector = create_tfidf_vector(dictionary, comment_list, tfidf)
print(f'Vetor TF-IDF: {tfidf_vector}')

# Encontrar a palavra na posicao que voce deseja, por exempro, na posicao 0 se usar a basse de dados que eu usei ira retornar hotel, porque e a primeira palavra
word = find_word(dictionary, 0)
# ^^^ Aqui eu supus que nao vao digitar numeros um pouco grandes, tipo 9.10^(10^100), porque se nao vai dar erro
print(f'Palavra na posicao 1: {word}')
print("\n")
print(dictionary)