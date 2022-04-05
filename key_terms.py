from lxml import etree
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

root = etree.parse('news.xml').getroot()
stop_words = stopwords.words('english') + ['ha', 'wa', 'u', 'a']
lemmatizer = WordNetLemmatizer()
all_list = []
heads_list = []


def doc_in_dataset():
    tokens_list = nltk.tokenize.word_tokenize(i[1].text.lower())
    tokens_list = [lemmatizer.lemmatize(token) for token in tokens_list]
    tokens_list = [a for a in tokens_list if a not in list(string.punctuation) and a not in stop_words]
    tokens_list = [nltk.pos_tag([x])[0][0] for x in tokens_list if nltk.pos_tag([x])[0][1] == 'NN']
    return ' '.join(tokens_list)


for i in root[0]:
    heads_list.append(i[0].text + ':')
    all_list.append(doc_in_dataset())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_list)
terms = vectorizer.get_feature_names_out()
matrix = tfidf_matrix.toarray()
index_dict = {}

for i, head in zip(range(len(matrix)), heads_list):
    print(head)
    for element, term in zip(matrix[i], terms):
        index_dict[term] = element
    sorted_tuples = sorted(index_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_dict = {k: v for k, v in sorted_tuples}
    res = sorted(sorted_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)[:5]
    print(*[a[0] for a in res], sep=' ')