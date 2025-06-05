import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from base_data import BaseData
from base_dataframe import BaseDataFrame
from model import Model

numbers = [i for i in range(0, 46)]

"""Получение датафрейма"""
base_data_frame = BaseDataFrame()
df = base_data_frame.df

"""Обработка текста"""
base_data = BaseData()

df['description'] = df.apply(lambda x: base_data.clean_text(x['description']), axis=1)

base_data.tokenize_text_list(df['description'])
base_data.delete_tokenize_stop_words()
base_data.lemmatize_tokenize_test()

df['text_stem'] = base_data.tokenize_texts_list

df['text_stem'] = base_data.tokenize_tests_list_lemmtize
X = df['text_stem']
y = df['class_cri']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

models = Model()

"""Классификатор Naive Bayes Classifier"""
nbs = models.get_naive_bayes_classifier_model()
nbs.fit(X_train, y_train)

y_pred = nbs.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
result_nbs = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))


"""Модель Linear Support Vector Machine"""
sgd = models.get_linear_support_vector_model()
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

result_sgd = classification_report(y_test, y_pred, output_dict=True)

"""Модель Logistic Regression"""

lr = models.get_logistic_regression_model()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

result_lr = classification_report(y_test, y_pred, output_dict=True)





# Графики основных показателей

# x = [1, 2, 3, 4]
# parametres = [
#     [result_nbs['1']['precision'], result_nbs['2']['precision'], result_nbs['3']['precision'], result_nbs['4']['precision']],
#     [result_nbs['1']['recall'], result_nbs['2']['recall'], result_nbs['3']['recall'], result_nbs['4']['recall']],
#     [result_nbs['1']['f1-score'], result_nbs['2']['f1-score'], result_nbs['3']['f1-score'], result_nbs['4']['f1-score']],
#     [result_nbs['1']['support'], result_nbs['2']['support'], result_nbs['3']['support'], result_nbs['4']['support']],
#     [result_sgd['1']['precision'], result_sgd['2']['precision'], result_sgd['3']['precision'], result_sgd['4']['precision']],
#     [result_sgd['1']['recall'], result_sgd['2']['recall'], result_sgd['3']['recall'], result_sgd['4']['recall']],
#     [result_sgd['1']['f1-score'], result_sgd['2']['f1-score'], result_sgd['3']['f1-score'], result_sgd['4']['f1-score']],
#     [result_sgd['1']['support'], result_sgd['2']['support'], result_sgd['3']['support'], result_sgd['4']['support']],
#     [result_lr['1']['precision'], result_lr['2']['precision'], result_lr['3']['precision'], result_lr['4']['precision']],
#     [result_lr['1']['recall'], result_lr['2']['recall'], result_lr['3']['recall'], result_lr['4']['recall']],
#     [result_lr['1']['f1-score'], result_lr['2']['f1-score'], result_lr['3']['f1-score'], result_lr['4']['f1-score']],
#     [result_lr['1']['support'], result_lr['2']['support'], result_lr['3']['support'], result_lr['4']['support']],
#
# ]
#
# plt.bar(x, parametres[0])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('Precision') #Подпись для оси y
# plt.title('Precision NBS') #Название
# plt.show()
#
# plt.bar(x, parametres[1])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('recall') #Подпись для оси y
# plt.title('recall NBS') #Название
# plt.show()
#
# plt.bar(x, parametres[2])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('f1-score') #Подпись для оси y
# plt.title('f1-score NBS') #Название
# plt.show()
#
# plt.bar(x, parametres[3])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('support') #Подпись для оси y
# plt.title('support NBS') #Название
# plt.show()
#
# plt.bar(x, parametres[4])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('Precision') #Подпись для оси y
# plt.title('Precision SGD') #Название
# plt.show()
#
# plt.bar(x, parametres[5])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('recall') #Подпись для оси y
# plt.title('recall SGD') #Название
# plt.show()
#
# plt.bar(x, parametres[6])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('f1-score') #Подпись для оси y
# plt.title('f1-score SGD') #Название
# plt.show()
#
# plt.bar(x, parametres[7])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('support') #Подпись для оси y
# plt.title('support SGD') #Название
# plt.show()
#
# plt.bar(x, parametres[8])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('Precision') #Подпись для оси y
# plt.title('Precision LR') #Название
# plt.show()
#
# plt.bar(x, parametres[9])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('recall') #Подпись для оси y
# plt.title('recall LR') #Название
# plt.show()
#
# plt.bar(x, parametres[10])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('f1-score') #Подпись для оси y
# plt.title('f1-score LR') #Название
# plt.show()
#
# plt.bar(x, parametres[11])
# plt.xlabel('Классы') #Подпись для оси х
# plt.ylabel('support') #Подпись для оси y
# plt.title('support LR') #Название
# plt.show()
#
# x_param = [1,2,3]
# y = [round(result_nbs['accuracy'], 2), round(result_sgd['accuracy'], 2), round(result_lr['accuracy'], 2)]
# print(len(y))
# print(len(x_param))
# plt.bar(x_param, y)
# plt.xlabel('Модели') #Подпись для оси х
# plt.ylabel('Точность') #Подпись для оси y
# plt.title('Accuracy models') #Название
# plt.show()