#импортим библиотеки
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3 as db
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import numpy as np
#расшифровка значений
dictt = {
    1: "Новость",
    2: "Реклама",
    3: "Мем",
    4: "Рекомендация",
    5: "Вопрос",
    6: "Отзыв",
    7: "История",
    8: "Мотивация",
    9: "Отчет",
    10: "Образование",
    11: "Инфографика",
    12: "Обсуждение",
    13: "Поздравления и благодарности",
    14: "Бэкстейдж",
    15: "Розыгрыш",
}



def append_data_to_existing_csv(file_name, data):
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data, columns=["text", "post_type"])
    
    # Добавляем данные в существующий файл
    df.to_csv(file_name, mode='a', index=False, header=False, encoding='utf-8')

new_data = [ [input(), '-'] ]

# Добавляем данные в CSV-файл
append_data_to_existing_csv('posts.csv', new_data)


data = pd.read_csv("posts.csv")
textss = data['text'].tolist()
y = data['post_type'].tolist()

#загружаем обученный конвейер для русского языка
nlp = spacy.load("ru_core_news_lg")

texts = []
stop_words = nlp.Defaults.stop_words
custom_stop_words = ["№", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "]", "{", "}", ";", ":", "'", "\"", "\\", "|", ",", ".", "<", ">", "/", "?", "~", "`"]
stop_words.update(custom_stop_words)
#очитска, нормализация и токенизация данных
for i in textss:
    s = ''
    doc = nlp(i)
    for token in doc:
        if not token.is_punct and not token.is_stop and not str(token).isdigit():
            s+=token.lemma_
            s+=' '
    texts.append(s)



unique_words = set(word for text in texts for word in text.split())
vocab = {word: i for i, word in enumerate(unique_words)}
def one_hot_encode(text):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            vector[vocab[word]] = 1
    return vector
x = np.array([one_hot_encode(text) for text in texts])
X_train, X_test, y_train , y_test = x[0:2500], x[0:2500], y[0:2500], y[0:2500]
    
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
print(f"Точность SVM: {accuracy_score(y_test, y_pred_svm)}")
# вывод и обработкак ответа
di = int(y_pred_svm[-1])
print(dictt.get(di))

# удаление данных, которые ввел пользователь
df = pd.read_csv('posts.csv')

# удаление строк, где значение в колонке 'столбец' равно 'значение'
df = df[df['post_type'] != '-']

# сохранение изменённого DataFrame в CSV-файл
df.to_csv('posts.csv', index=False)
