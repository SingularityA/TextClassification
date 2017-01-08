import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


def clean_data(text):
    stemmer = SnowballStemmer('russian')
    stop_words = stopwords.words('russian')

    data = re.sub('\W', ' ', text).split()
    return ' '.join([stemmer.stem(word) for word in data if word not in stop_words])


def extract_data(path, limit=61000):
    labels, contents = [], []
    counter = 0
    for line in open(path, encoding='utf8'):
        if counter == limit:
            break
        label, content = line.split('\t', maxsplit=1)
        labels.append(label)
        contents.append(clean_data(content))

        counter += 1
    return labels, contents


labels, contents = extract_data('news/news_train.txt')

# вместе с заголовком
train_data = contents
print("Reading finished")

# векторизация
tfid_vectorizer = TfidfVectorizer(max_features=31000, norm='l1')
train_vect_data = tfid_vectorizer.fit_transform(train_data)
print("Vectorized:", train_vect_data.shape)

# тренировка
# параметр C подобран эмпирически на меньших данных
clf = svm.LinearSVC(C=41., dual=False, verbose=1).fit(train_vect_data, labels)
print("Trained")

# чтение тестовых данных из файла
test_data = []
for line in open('news/news_test.txt', encoding='utf8'):
    test_data.append(clean_data(line))

# векторизация тестовых данных
test_vect_data = tfid_vectorizer.transform(test_data)

# предсказание
predicted_labels = clf.predict(test_vect_data)
print("Predicted")

# вывод результата в файл
with open('news/news_output.txt', 'w+', encoding='utf8') as fout:
    fout.write('\n'.join(predicted_labels))
