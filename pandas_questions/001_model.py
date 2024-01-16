import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# df = load the df


# One-hot encode 'level' column
encoder = OneHotEncoder()
level_encoded = encoder.fit_transform(df[['level']]).toarray()
level_df = pd.DataFrame(level_encoded, columns=encoder.get_feature_names(['level']))
df = pd.concat([df, level_df], axis=1).drop(['level'], axis=1)

# Define features (X) and target (y)
X = df.drop(['course_id', 'course_title', 'is_popular'], axis=1)
y = df['is_popular']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))



==========================================================================
NLP text classification
==========================================================================

import string
import nltk
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy resources
nlp = spacy.load('en_core_web_sm')

# Sample data (replace this with your dataset)
texts = ["This is a good sentence.", "This is a bad sentence, a defaulter sentence with bail-out!"]
labels = [1, 0]  # Example labels

# 1. Text Normalization
# a. Stop words removal
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# b. Tokenization, c. Stemming, d. Lemmatization, e. Case-Folding
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def normalize_text(text):
    # Tokenization and case-folding
    tokens = nltk.word_tokenize(text.lower())
    # Stop words removal
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    # Lemmatization
    lemmatized_tokens = [nlp(token)[0].lemma_ for token in stemmed_tokens]
    return ' '.join(lemmatized_tokens)

# Apply normalization
normalized_texts = [normalize_text(text) for text in texts]

# 2. Text Standardization
jargon_list = ['defaulter', 'bail-out']

def standardize_text(text):
    # Jargon removal
    no_jargon_text = ' '.join([word for word in text.split() if word not in jargon_list])
    # Remove grammatical mistakes (simple approach, may need a more sophisticated method)
    standardized_text = ' '.join([word for word in no_jargon_text.split() if word.isalpha()])
    return standardized_text

# Apply standardization
standardized_texts = [standardize_text(text) for text in normalized_texts]

# Part-of-Speech tagging
pos_tags = [nltk.pos_tag(nltk.word_tokenize(text)) for text in standardized_texts]

# Feature Extraction (you might want to use TfidfVectorizer or other methods as well)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(standardized_texts)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Machine Learning Model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print POS tags
print("POS Tags:", pos_tags)
