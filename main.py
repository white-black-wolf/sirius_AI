#коннектим библиотеки
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sqlite3 as db
from sklearn.svm import SVC

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

#коннектим датасет
conn = db.connect('post_types_copy.db')
cursor = conn.cursor()
a = [(input(), '-')] # обрабатываем ввод пользователя
cursor.executemany("INSERT INTO posts (text, post_type) VALUES (?, ?)", a) #добавляем в датасет
conn.commit()
# получаем значения из датасета
conn.row_factory = lambda cursor, row: row[0]
c = conn.cursor()
texts = c.execute('SELECT text FROM posts').fetchall()
y = c.execute('SELECT post_type FROM posts').fetchall()

#получаем ветор
vectorizer = CountVectorizer()#иниацилизация
X = vectorizer.fit_transform(texts)
x = X.toarray()
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Пример меток классов

X_train, X_test, y_train , y_test = x[0:975], x[975:], y[0:975], y[975:]


nb_classifier = MultinomialNB()#инициализация
nb_classifier.fit(X_train, y_train)#обучаем модель
y_pred = nb_classifier.predict(X_test)#получаем ответ от нейросети
di = int(y_pred[-1])#обрабатываем ответ
print(dictt.get(di))#выводим ответ

#удаляем последнее значение
cursor.execute(f'DELETE FROM posts WHERE id = {len(x)}')
conn.commit()
conn.close()