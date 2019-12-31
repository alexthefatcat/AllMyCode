# -*- coding: utf-8 -*-"""Created on Thu Jul  5 14:06:44 2018@author: milroa1"""


article ="""
The Ministry of Truth, which concerned itself with news, entertainment, education and the fine arts.
The Ministry of Peace, which concerned itself with war. The Ministry of Love, which maintained law
and order. And the Ministry of Plenty, which was responsible for economic affairs. Their names, in
Newspeak: Minitrue, Minipax, Miniluv and Miniplenty.

Like Olympic medals and tennis trophies, all they signified was that the owner had done something of
no benefit to anyone more capably than everyone else"""

    
#%%###############################################################################################################################     
"""                                            Natural Language Processing                                                                   """
##################################################################################################################################
"""
gensim-spacy-nltk

What is NLP ?
  natural language processing (NLP) is about developing applications and services that are able to understand human languages.
  We are talking here about practical examples of natural language processing (NLP) like speech recognition, speech translation,
  understanding complete sentences, understanding synonyms of matching words, and writing complete grammatically correct sentences and paragraphs.

What is Tokenization?
  chops article into pieces called tokens often removing non-alpha characters,
  remember these can be words or senteces but for senteces remember that MR. and other things
  have a full stop nltk is smart and avoids the exceptions

What is a stop word?
  generally words that carry little information like:- "the", "and"
  removing them makes processing easier

What is bag-of-words ?
  counts the number of words in a text
    or normalized // Term Frequency-Inverse Document Frequency (TF-IDF)> word_count_in_doc * log(sum(docs)/docs_that_contain_word)

What is named entity recognition ?
  shows the important word like dates location people

What is corpus ?

What is a Synonym?,Antonyms
A synonym is a word or phrase that means exactly or nearly the same as another word or phrase, antonum(is the opposite short>tall). 

What is Stemming ?
  Word stemming means removing affixes from words and returning the root word. ( [working,worked,works,work] > work) 
  PorterStemmer is a popular algorthium, NLTK uses this however it's not perfect ( increases> increas).
  lemmatizing is a smarter way to do this

What is Lemmatizing ?
  Word lemmatizing is similar to stemming, but the difference is the result of lemmatizing is a real word. 
  ( [working,worked,works,work] > work)    ( increases> increase)

What is Part of Speech Tagging ?
  a list of words are input and each of the word is labelled life labels like noun verb...etc

#What is Chunking(also called shallow parsing)
#  Chunking  is the process by which we group various words together by their part of speech tags. 
#https://www.youtube.com/watch?v=imPpT2Qo2sk&index=5&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL



What is a Word Vector
ngram


chunking
chinking
nltk corpora
wordnet
text classification
words as features for learning
naive bayes
Sentiment analysis
word embedding

## neaural networks + machine learning

Word (or Character, Sentence, Document) Embeddings.




a neural probabilistic language model
word2vec is a software
2 algorthiums
skip-grams: predict context words given tarjet
continuous bag of words:predict target word from bag-of-words context

skip-gram prediction
uses word vectors
predicting the surrounding words around the center word
                  _______
... "turning into banking crises as" ...
                  ¯¯¯¯¯¯¯            


"""
##################################################################################################################################
import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize  # for words
from nltk.tokenize import sent_tokenize  # for sentences
from nltk.corpus   import wordnet        # really useful package all though has defintion for every words
from nltk.stem     import PorterStemmer
from nltk.stem     import WordNetLemmatizer

stemmer    = PorterStemmer() 
lemmatizer = WordNetLemmatizer()
stopwords_ = stopwords.words('english')

from collections   import Counter

##################################################################################################################################


#basic word tokenize
tokens1 = [t for t in article.split()] 

# advanced remove question marks etc and lower all characrters# isalpha removes non alphabetstiff
tokens0 = Counter( [t.lower() for t in word_tokenize(article) if t.isalpha()] )

#remove stopwords
tokens2 = [t for t in tokens1 if t not in stopwords_ ]




freq = nltk.FreqDist(tokens2) 
for key,val in freq.items(): 
    print (str(key) + ':' + str(val))

##################################################################################################################################
"""   WordNet Usefull Stuff   """

syn = wordnet.synsets("pain")

#synonyms i.e. similar words to the word 'Computer'
synonyms = [ lemma.name()               for lemma in syn.lemmas() for syn in wordnet.synsets('Computer')                     ]
antonyms = [ lemma.antonyms()[0].name() for lemma in syn.lemmas() for syn in wordnet.synsets('small'   ) if lemma.antonyms() ]#>#['large', 'big', 'big']


stem=stemmer.stem('working')#># stem = 'work'


##################################################################################################################################




#####################################
#gensim

total_word_count = defaultdict(int)

for word_id, word_count in itertools.chain.from_iterable(corpus):

    total_word_count[word_id] += word_count

sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 
######################################################





## part_of_speech_agging
"creates a list of tupples where word and type(verb noun..etc)"
def part_of_speech_tagger__for_list_of_words(word_list):
    nltk.help.upenn_tagset()#print the tag translation NN is like Noun
    try:
            tagged = nltk.pos_tag(word_list)
    except Exception as e:
       print(str(e))
    return( tagged )

nested_tokenize = [nltk.word_tokenize(sentence) for sentence in tokenized ]
tokenized_pos = [part_of_speech_tagger__for_list_of_words(word_list) for word_list in nested_tokenize]


"""
what are the following :-
 (BoW), Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA) etc

pre-trained word embedding models which including Word2Vec
"""




# article
article_tok_sw =[word_tokenize(sent) for sent in sent_tokenize(article) ]





# chunk creates a nlp tree thing




# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)
# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1         
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())
# Create a list of the values: values
values = [ner_categories.get(l) for l in labels]
# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
# Display the chart
plt.show()
#%%##########################################################
"""                    spacy                              """
#############################################################
# Import spacy
import spacy
# Instantiate the English model: nlp
nlp = spacy.load("en",tagger=False, parser=False, matcher=False)
# Create a new document: doc
doc = nlp(article)
# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
#%%##########################################################
#polyglot good for translation-- I dont really care about this
entities = [ (ent.tag,' '.join(ent))  for ent in txt.entities]
# Print entities
print(entities)


#%%##########################################################
"""   skikit-learn            """
import pandas as pd
from skylearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CounterVectorizer
df ... #load data
y = df["Sci-Fi"]
Xte,Yte,Xtr,Ytr =




# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

































