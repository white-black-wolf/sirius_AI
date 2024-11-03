<p align="center">
 <img src="https://i.ibb.co/99qgRP3/summma.jpg" alt="Project logo"width="726">
</p>
<a id="up"></a>
<p align="center">
 <img src="https://img.shields.io/badge/python-3.11-blue" alt="Версия python">
 <img src="https://img.shields.io/badge/version-0.1(beta)-purple" alt="Версия игры">
 <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="Лицензия">
</p>

## 📑Оглавление
- [Описание задачи](#task)
- [O проеке](#about_project)
- [Рабочий процесс](#process)
- [Документация](#documentation)
- [Разработчики](#developers)
- [Лицензия](#license)

<a id="task"></a>
## 📖Описание задачи
На вход программе подаётся текст. Результатом работы программы должен стать тип поста в котором мог быть этот текст!


<a id="about_project"></a>
## 📁О проекте
Мы разработали **Классификатор постов в социальной сети ВКонтакте**.
Это система, предназначенная для автоматической классификации контента, размещаемого пользователями. Его основная задача заключается в анализе и сортировке постов по темам.

Классификатор использует алгоритмы машинного обучения и обработки естественного языка, чтобы определить содержание постов и их назначение. Например, он может отличать посты, относящиеся к новостям, развлекательному контенту, образовательным материалам и т.д. Это позволяет пользователям быстрее находить интересующий их контент и улучшает качество взаимодействия в социальной сети.

Кроме того, такая система может быть полезна для рекламодателей, которые хотят таргетировать свою аудиторию более точно, выбирая посты, соответствующие определённой категории или интересам пользователей. В результате, классификатор постов в ВКонтакте способствует более организованному и эффективному контенту на платформе.


<a id="process"></a>
## ✍️Рабочий процесс
- Сбор необходимых данных.
-  Предварительная обработка данных.
-  Выбор и обучение модели.
-  Оценка решения.
-   Презентация решения.


<a id="documentation"></a>
## 🗃️Документация

Данный проект создан с целью определения типов постов в социальных сетях! Для этого мы использовали модель машинного обучения MultinomialNB из библиотеки [sklearn](https://scikit-learn.org/stable/index.html).

Multinomial Naive Bayes (MultinomialNB) — это мощный алгоритм, известный своей простотой и эффективностью в задачах классификации текстов. Он представляет собой вероятностный алгоритм, основанный на [теореме Байеса](https://ru.wikipedia.org/wiki/Теорема_Байеса), с допущением о независимости между характеристиками.
Алгоритм MultinomialNB предназначен для обработки дискретных данных. Он широко используется в задачах обработки естественного языка, таких как фильтрация спама, анализ настроения и категоризация документов.

Для анализа текста с помощью модели машинного обучения мы преобразововали каждое слово в тексте в вектор с помощью метода [Bag-of-Words](https://ru.wikipedia.org/wiki/Мешок_слов).


<a id="developers"></a>
## 👨🏻‍💻Разработчики

- [wbw](https://github.com/white-black-wolf)
- [Vanish](https://github.com/vanish12345)   

<a id="license"></a>
## 🏛️Лицензия
Проект sirius_AI распространяеться под лицензией MIT.

 [Лицензия](https://github.com/white-black-wolf/sirius_AI/blob/main/LICENSE)
 
<br></br>

 [Вверх](#up)
