from __future__ import division #To avoid integer division
from operator import itemgetter
###Training Phase###

f = open("./Datasets/code_mixed_train.txt","r")
tr_str = ""
x = f.readlines()
for i in x:
  st = i.split()
  
  if st:
    tr_str += st[0] + "/" + st[-2] + "\n"

tr_li = tr_str.split()

j = 0
s = set()
while j < len(tr_li):
  classify = tr_li[j].split("/")[1] 
  if classify and classify not in ['G_PRP', 'G_N', 'G_V', 'G_J', 'G_X', 'G_R', 'G_PRT', 'G_SYM']:
    tr_li.pop(j)
  else:
    j += 1 
print(tr_li)

num_words_train = len(tr_li)

print(num_words_train)

train_li_words = ['']

train_li_words*= num_words_train

train_li_tags = ['']
train_li_tags*= num_words_train

for i in range(num_words_train):
    temp_li = tr_li[i].split("/")
    train_li_words[i] = temp_li[0]
    train_li_tags[i] = temp_li[1]


#print(len(train_li_words), len(train_li_tags))

dict2_tag_follow_tag_ = {}
"""Nested dictionary to store the transition probabilities
each tag A is a key of the outer dictionary
the inner dictionary is the corresponding value
The inner dictionary's key is the tag B following A
and the corresponding value is the number of times B follows A
"""

dict2_word_tag = {}
"""Nested dictionary to store the emission probabilities.
Each word W is a key of the outer dictionary
The inner dictionary is the corresponding value
The inner dictionary's key is the tag A of the word W
and the corresponding value is the number of times A is a tag of W
"""

dict_word_tag_baseline = {}
#Dictionary with word as key and its most frequent tag as value

for i in range(num_words_train-1):
    outer_key = train_li_tags[i]
    inner_key = train_li_tags[i+1]
    dict2_tag_follow_tag_[outer_key]=dict2_tag_follow_tag_.get(outer_key,{})
    dict2_tag_follow_tag_[outer_key][inner_key] = dict2_tag_follow_tag_[outer_key].get(inner_key,0)
    dict2_tag_follow_tag_[outer_key][inner_key]+=1

    outer_key = train_li_words[i]
    inner_key = train_li_tags[i]
    dict2_word_tag[outer_key]=dict2_word_tag.get(outer_key,{})
    dict2_word_tag[outer_key][inner_key] = dict2_word_tag[outer_key].get(inner_key,0)
    dict2_word_tag[outer_key][inner_key]+=1


"""The 1st token is indicated by being the 1st word of a senetence, that is the word after period(.)
Adjusting for the fact that the first word of the document is not accounted for that way
"""

dict2_tag_follow_tag_['.'] = dict2_tag_follow_tag_.get('.',{})
dict2_tag_follow_tag_['.'][train_li_tags[0]] = dict2_tag_follow_tag_['.'].get(train_li_tags[0],0)
dict2_tag_follow_tag_['.'][train_li_tags[0]]+=1


last_index = num_words_train-1

#Accounting for the last word-tag pair
outer_key = train_li_words[last_index]
inner_key = train_li_tags[last_index]
dict2_word_tag[outer_key]=dict2_word_tag.get(outer_key,{})
dict2_word_tag[outer_key][inner_key] = dict2_word_tag[outer_key].get(inner_key,0)
dict2_word_tag[outer_key][inner_key]+=1


"""Converting counts to probabilities in the two nested dictionaries
& also converting the nested dictionaries to outer dictionary with inner sorted lists
"""
for key in dict2_tag_follow_tag_:
    di = dict2_tag_follow_tag_[key]
    s = sum(di.values())
    for innkey in di:
        di[innkey] /= s
    di = di.items()
    di = sorted(di,key=lambda x: x[0])
    dict2_tag_follow_tag_[key] = di

for key in dict2_word_tag:
    di = dict2_word_tag[key]
    dict_word_tag_baseline[key] = max(di, key=di.get)
    s = sum(di.values())
    for innkey in di:
        di[innkey] /= s
    di = di.items()
    di = sorted(di,key=lambda x: x[0])
    dict2_word_tag[key] = di



###Testing Phase###    


f1 = open("./Datasets/code_mixed_test.txt","r")
te_str = ""
x = f1.readlines()
for i in x:
  st = i.split()
  
  if st:
    te_str += st[0] + "/" + st[-2] + "\n"
te_li = te_str.split()

j = 0
s = set()
while j < len(te_li):
  classify = te_li[j].split("/")[1] 
  if classify and classify not in ['G_PRP', 'G_N', 'G_V', 'G_J', 'G_X', 'G_R', 'G_PRT', 'G_SYM']:
    te_li.pop(j)
  else:
    j += 1 

print(te_li)

num_words_test = len(te_li)

test_li_words = ['']
test_li_words*= num_words_test

test_li_tags = ['']
test_li_tags*= num_words_test

output_li = ['']
output_li*= num_words_test

output_li_baseline = ['']
output_li_baseline*= num_words_test

num_errors = 0
num_errors_baseline = 0

for i in range(num_words_test):
    temp_li = te_li[i].split("/")
    test_li_words[i] = temp_li[0]
    test_li_tags[i] = temp_li[1]

    output_li_baseline[i] = dict_word_tag_baseline.get(temp_li[0],'')
    #If unknown word - tag = 'NNP'
    if output_li_baseline[i]=='':
        output_li_baseline[i]='G_N'
        
        


    if output_li_baseline[i]!=test_li_tags[i]:
        num_errors_baseline+=1

    
    if i==0:    #Accounting for the 1st word in the test document for the Viterbi
        di_transition_probs = dict2_tag_follow_tag_['.']
    else:
        di_transition_probs = dict2_tag_follow_tag_[output_li[i-1]]
        
    di_emission_probs = dict2_word_tag.get(test_li_words[i],'')

    #If unknown word  - tag = 'NNP'
    if di_emission_probs=='':
        output_li[i]='G_N'
        
    else:
        max_prod_prob = 0
        counter_trans = 0
        counter_emis =0
        prod_prob = 0
        while counter_trans < len(di_transition_probs) and counter_emis < len(di_emission_probs):
            tag_tr = di_transition_probs[counter_trans][0]
            tag_em = di_emission_probs[counter_emis][0]
            if tag_tr < tag_em:
                counter_trans+=1
            elif tag_tr > tag_em:
                counter_emis+=1
            else:
                prod_prob = di_transition_probs[counter_trans][1] * di_emission_probs[counter_emis][1]
                if prod_prob > max_prod_prob:
                    max_prod_prob = prod_prob
                    output_li[i] = tag_tr
                    #print "i=",i," and output=",output_li[i]
                counter_trans+=1
                counter_emis+=1    
    

    if output_li[i]=='': #In case there are no matching entries between the transition tags and emission tags, we choose the most frequent emission tag
        output_li[i] = max(di_emission_probs,key=itemgetter(1))[0]  
        
    if output_li[i]!=test_li_tags[i]:
        num_errors+=1

                    
print("FAccuracy (Baseline) :",((num_words_test - num_errors_baseline)/num_words_test))
print("Accuracy (Viterbi):",((num_words_test - num_errors)/num_words_test))

#print "Tags suggested by Baseline Algorithm:", output_li_baseline

#print "Tags suggested by Viterbi Algorithm:", output_li

#print "Correct tags:",test_li_tags

bcount=0
bcount1=0

vcount=0
vcount1=0

print(len(test_li_tags))
print(len(output_li))

for i in range(0,len(test_li_tags)):
    if(output_li_baseline[i]==test_li_tags[i]):
        bcount+=1
    else:
        bcount1+=1
    if(output_li[i]==test_li_tags[i]):
        vcount+=1
    else:
        vcount1+=1
'''  
print(test_li_tags)
print(output_li)
print(len(test_li_tags))
print(len(output_li))
print(bcount,bcount1)
print(vcount, vcount1)
''' 








from __future__ import division  # To avoid integer division
from operator import itemgetter
import numpy as np

###Training Phase###

f = open("./Datasets/code_mixed_train.txt", "r")
tr_str = ""
x = f.readlines()
for i in x:
    st = i.split()
    if st:
        tr_str += st[0] + "/" + st[-2] + "\n"

tr_li = tr_str.split()

j = 0
s = set()
while j < len(tr_li):
    classify = tr_li[j].split("/")[1]
    if classify and classify not in ['G_PRP', 'G_N', 'G_V', 'G_J', 'G_X', 'G_R', 'G_PRT', 'G_SYM']:
        tr_li.pop(j)
    else:
        j += 1

num_words_train = len(tr_li)

train_li_words = ['']
train_li_words *= num_words_train

train_li_tags = ['']
train_li_tags *= num_words_train

for i in range(num_words_train):
    temp_li = tr_li[i].split("/")
    train_li_words[i] = temp_li[0]
    train_li_tags[i] = temp_li[1]


###Testing Phase###

f1 = open("./Datasets/code_mixed_test.txt", "r")
te_str = ""
x = f1.readlines()
for i in x:
    st = i.split()
    if st:
        te_str += st[0] + "/" + st[-2] + "\n"
te_li = te_str.split()

j = 0
s = set()
while j < len(te_li):
    classify = te_li[j].split("/")[1]
    if classify and classify not in ['G_PRP', 'G_N', 'G_V', 'G_J', 'G_X', 'G_R', 'G_PRT', 'G_SYM']:
        te_li.pop(j)
    else:
        j += 1

num_words_test = len(te_li)

test_li_words = ['']
test_li_words *= num_words_test

test_li_tags = ['']
test_li_tags *= num_words_test

output_li = ['']
output_li *= num_words_test

hmm_predictions = []
hmm_probs = []

for i in range(num_words_test):
    temp_li = te_li[i].split("/")
    test_li_words[i] = temp_li[0]
    test_li_tags[i] = temp_li[1]

    if i == 0:  
        di_transition_probs = dict2_tag_follow_tag_['.']
    else:
        di_transition_probs = dict2_tag_follow_tag_[output_li[i - 1]]

    di_emission_probs = dict2_word_tag.get(test_li_words[i], '')

    # If unknown word - tag = 'G_N'
    if di_emission_probs == '':
        output_li[i] = 'G_N'
        hmm_predictions.append('G_N')
        hmm_probs.append(0.0)  # Assign a probability of 0.0 for unknown words

    else:
        max_prod_prob = 0
        counter_trans = 0
        counter_emis = 0
        prod_prob = 0
        while counter_trans < len(di_transition_probs) and counter_emis < len(di_emission_probs):
            tag_tr = di_transition_probs[counter_trans][0]
            tag_em = di_emission_probs[counter_emis][0]
            if tag_tr < tag_em:
                counter_trans += 1
            elif tag_tr > tag_em:
                counter_emis += 1
            else:
                prod_prob = di_transition_probs[counter_trans][1] * di_emission_probs[counter_emis][1]
                if prod_prob > max_prod_prob:
                    max_prod_prob = prod_prob
                    output_li[i] = tag_tr
                counter_trans += 1
                counter_emis += 1

        if output_li[i] == '':  # In case there are no matching entries between the transition tags and emission tags
            output_li[i] = max(di_emission_probs, key=itemgetter(1))[0]

        hmm_predictions.append(output_li[i])
        hmm_probs.append(max_prod_prob)

# Convert predictions and probabilities to the required format
labels = {'G_PRP': 0, 'G_N': 1, 'G_V': 2, 'G_J': 3, 'G_X': 4, 'G_R': 5, 'G_PRT': 6, 'G_SYM': 7, "ne": 8}
hmm_predictions = [labels[tag] for tag in hmm_predictions]
hmm_probs = np.array([np.eye(len(labels))[label] for label in hmm_predictions])

# Print the baseline accuracy and Viterbi accuracy
num_errors_baseline = sum(1 for i in range(num_words_test) if dict_word_tag_baseline.get(test_li_words[i], '') != test_li_tags[i])
num_errors = sum(1 for i in range(num_words_test) if output_li[i] != test_li_tags[i])
print("Accuracy (Baseline):", ((num_words_test - num_errors_baseline) / num_words_test))
print("Accuracy (Viterbi):", ((num_words_test - num_errors) / num_words_test))


#---------------------------------------------------------------------------------------------------------------------


import nltk
import sklearn_crfsuite
import pycrfsuite
import itertools
import numpy as np
from sklearn.metrics import classification_report

file = open('/content/ICON_Coarse_Train.txt')
file_data = file.readlines()
file1 = open('/content/ICON_Coarse_Test.txt')
file_data1 = file1.readlines()
train_data = []
train = []
test_data = []
test = []

for i in range(0, len(file_data)):
    if len(file_data[i].split()) > 0:
        m = tuple(file_data[i].split())
        train.append(m)
    else:
        train_data.append(train)
        train = []

for i in range(0, len(file_data1)):
    if len(file_data1[i].split()) > 0:
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
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][2]
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
    return [postag for token, lang, postag in sent]

def sent2tokens(sent):
    return [token for token, lang, postag in sent]

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

crf_predictions = []
crf_probs = []

for xseq in X_test:
    # Get the predicted tags and their probabilities
    tags, probs = tagger.tag(xseq, marginals=True)
    crf_predictions.extend(tags)
    crf_probs.extend(probs)

# Create a mapping of POS tags to indices
pos_tags = set(itertools.chain.from_iterable(y_test))
pos_tags_to_idx = {tag: i for i, tag in enumerate(pos_tags)}

# Convert the predictions to numerical indices
crf_predictions = [pos_tags_to_idx[tag] for tag in crf_predictions]

# Convert the probabilities to the required format
crf_probs = np.array([np.eye(len(pos_tags_to_idx))[pos_tags_to_idx[tag]] for tag in crf_predictions for _ in range(len(xseq))])

# Print out the classification report
truths = np.array([pos_tags_to_idx[tag] for row in y_test for tag in row])
print(classification_report(truths, crf_predictions, target_names=list(pos_tags_to_idx.keys())))

# Print predictions and probabilities for a sample sentence
sentence = "Nenu ee cinema chusanu, chala bagundi. Andariki chudalani naa korika."
for i in sentence.split():
    print(i)
    print(tagger.tag([i]))


# ------------------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score

# Assuming you have the predictions and probabilities from the HMM and CRF models
# as hmm_predictions, hmm_probs, crf_predictions, and crf_probs

# Probability Voting
def probability_voting(hmm_probs, crf_probs):
    ensemble_probs = (hmm_probs + crf_probs) / 2
    ensemble_predictions = np.argmax(ensemble_probs, axis=1)
    return ensemble_predictions

# Rank-Based Voting
def rank_based_voting(hmm_predictions, crf_predictions):
    ensemble_predictions = []
    for hmm_pred, crf_pred in zip(hmm_predictions, crf_predictions):
        hmm_rank = np.argsort(-hmm_pred)
        crf_rank = np.argsort(-crf_pred)
        rank_sum = hmm_rank + crf_rank
        ensemble_pred = np.argmin(rank_sum)
        ensemble_predictions.append(ensemble_pred)
    return np.array(ensemble_predictions)

# Assuming you have the true labels as y_true
y_true = np.array([labels[tag] for row in y_test for tag in row])

accuracy_hmm = accuracy_score(y_true, hmm_predictions)
accuracy_crf = accuracy_score(y_true, crf_predictions)

# Probability Voting
ensemble_predictions_probs = probability_voting(hmm_probs, crf_probs)
accuracy_probs_voting = accuracy_score(y_true, ensemble_predictions_probs)

# Rank-Based Voting
ensemble_predictions_rank = rank_based_voting(hmm_probs, crf_probs)
accuracy_rank_voting = accuracy_score(y_true, ensemble_predictions_rank)

print("Accuracy (HMM):", accuracy_hmm)
print("Accuracy (CRF):", accuracy_crf)
print("Accuracy (Probability Voting):", accuracy_probs_voting)
print("Accuracy (Rank-Based Voting):", accuracy_rank_voting)