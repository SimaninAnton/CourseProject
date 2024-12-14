from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from base_data import BaseData
from base_dataframe import BaseDataFrame
from model import Model


"""Получение датафрейма"""
base_data_frame = BaseDataFrame()
df = base_data_frame.df

"""Обработка текста"""
base_data = BaseData()

df['description'] = df.apply(lambda x: base_data.clean_text(x['description']), axis=1)
base_data.tokenize_text_list(df['description'])

df['text_stem'] = base_data.tokenize_texts_list


X = df['text_stem']
y = df['class_cri']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state = 42)

models = Model()

"""Классификатор Naive Bayes Classifier"""
nbs = models.get_naive_bayes_classifier_model()
nbs.fit(X_train, y_train)

y_pred = nbs.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


"""Модель Linear Support Vector Machine"""
sgd = models.get_linear_support_vector_model()
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


"""Модель Logistic Regression"""

lr = models.get_logistic_regression_model()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
