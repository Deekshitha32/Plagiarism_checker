#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.metrics import edit_distance

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return stemmed_tokens

def calculate_similarity(text1, text2):
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    common_tokens = set(tokens1).intersection(tokens2)
    jaccard_similarity = len(common_tokens) / (len(set(tokens1)) + len(set(tokens2)) - len(common_tokens))
    
    return jaccard_similarity

def is_plagiarized(text1, text2, threshold=0.5):
    similarity = calculate_similarity(text1, text2)
    return similarity >= threshold

text1 = "This is an original and unique content."
text2 = "This is a completely different piece of writing."
text3 = "This is an original text with some additional content."

print("Text 1 and Text 2 similarity:", calculate_similarity(text1, text2))
print("Text 1 and Text 3 similarity:", calculate_similarity(text1, text3))

if is_plagiarized(text1, text2):
    print("Text 1 is plagiarized from Text 2.")
else:
    print("Text 1 is not plagiarized from Text 2.")

