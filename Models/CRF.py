import nltk
import sklearn_crfsuite
import pycrfsuite
import itertools
import numpy as np
from sklearn.metrics import classification_report

file = open('./Datasets/ICON_Coarse_Train.txt')
file_data = file.readlines()
file1 = open('./Datasets/ICON_Coarse_Test.txt')
file_data1 = file1.readlines()
train_data = []
train = []
test_data = []
test = []
for i in range(0,len(file_data)):
       if(len(file_data[i].split())>0):       
              m = tuple(file_data[i].split())
              train.append(m)
       else:
              train_data.append(train)
              train = []
for i in range(0,len(file_data1)):
       if(len(file_data1[i].split())>0):       
              m = tuple(file_data1[i].split())
              test.append(m)
       else:
              test_data.append(test)
              test = []

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [postag for token, lang ,postag  in sent]

def sent2tokens(sent):
    return [token for token,lang ,postag in sent]

X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

X_test = [sent2features(s) for s in test_data]
y_test = [sent2labels(s) for s in test_data]

trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 0.1,
    'c2': 0.01,
    'max_iterations': 200,
    'feature.possible_transitions': True
})

trainer.train('crf.model')

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Create a mapping of POS tags to indices
pos_tags = set(itertools.chain.from_iterable(y_test))
pos_tags_to_idx = {tag: i for i, tag in enumerate(pos_tags)}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([pos_tags_to_idx[tag] for row in y_pred for tag in row])
truths = np.array([pos_tags_to_idx[tag] for row in y_test for tag in row])

# Print out the classification report
print(classification_report(
    truths, predictions,
    target_names=list(pos_tags_to_idx.keys())))

sentence="Nenu ee cinema chusanu, chala bagundi. Andariki chudalani naa korika."
for i in sentence.split():
  print(i)
  print(tagger.tag([i]))
#print(X_test[0])