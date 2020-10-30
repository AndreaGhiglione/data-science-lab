import csv
import string
import math

def tokenize(docs):
    """Compute the tokens for each document.
    Input: a list of strings. Each item is a document to tokenize.
    Output: a list of lists. Each item is a list containing the tokens of the
    relative document.
    """
    tokens = []
    for doc in docs:
        for punct in string.punctuation:
            doc = doc.replace(punct, " ")
        split_doc = [ token.lower() for token in doc.split(" ") if token ]
        tokens.append(split_doc)
    return tokens


print('\nEX 2.1')
# Uploading the dataset as a list of lists
dataset = []
with open('aclimdb_reviews_train.csv',encoding='utf8') as file:
    next(csv.reader(file))  # I skip the first line of the csv
    dataset = list(csv.reader(file))
    reviews = list(map(lambda d: d[0], dataset))  # Or reviews = [d[0] for d in dataset]
    print(reviews)

print('\nEX 2.2')
reviews_tokenized = tokenize(reviews)  # A list of lists with the tokens for each document
print(reviews_tokenized)


print('\nEX 2.3')
def term_frequency(tokens):
    tf = []  # It will be a list of dictionaries
    for document in tokens:  # Document is a review in our case
        dictionary = {}
        for word in document:
            if word in dictionary:
                dictionary[word] += 1  # If the word is already in the dictionary I update the counter for that key
            else:
                dictionary[word] = 1  # Otherwise I add the new word to the dictionary with occurence 1 (first)
        tf.append(dictionary)  # I add the current dictionary (referred to the current document) at the total list
    return tf

list_words_counted = term_frequency(reviews_tokenized)
print(list_words_counted)


print('\nEX 2.4')
DFt = {}  # A dictionary which will contain the words of the reviews and the number of documents in which they appear
for words_counted in list_words_counted:
    for word in words_counted.keys():
        if word not in DFt:
            DFt[word] = 1
        else:
            DFt[word] += 1
print(DFt)

IDFt = {}
N = len(reviews)
for word in DFt.keys():  # keys are the words in the reviews (no repetitions)
    IDFt[word] = round(math.log2(N/DFt[word]),3)
print(IDFt)
sorted_IDFt = sorted(IDFt.items(), key=lambda x:x[1], reverse=False)
print(f'\n{sorted_IDFt[0:5]}')
print('As we can see the most common words are words very used in many speeches')


print('\nEX2.5')
TFIDFt = []
for words_counted in list_words_counted:
    dictionary = {}  # An empty dictionary which will contain the TFIDFt of the single review
    for word in words_counted.keys():
        dictionary[word] = round(words_counted[word] * IDFt[word],3)
    TFIDFt.append(dictionary)
print(TFIDFt)