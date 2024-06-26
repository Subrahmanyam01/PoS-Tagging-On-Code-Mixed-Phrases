from __future__ import division #To avoid integer division
from operator import itemgetter
###Training Phase###

f = open("./Datasets/code_mixed_train","r")
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


f1 = open("./Datasets/code_mixed_test","r")
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






