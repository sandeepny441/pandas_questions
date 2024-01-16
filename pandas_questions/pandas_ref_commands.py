# Numerical EDA:

sns.histplot(df[col])
sns.kdeplot(df[col])
df[col].kurtosis()
df[col].skew()
sns.boxplot(df[col])
=============================================
# OUTLIERS
q1 = df[col].quantile(25)
# Step 1: Calculate Q1 and Q3

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)

# Step 2: Compute IQR
IQR = Q3 - Q1

# Step 3: Determine the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Identify outliers
outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

# If you just want to filter out the outliers
filtered_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

=============================================
corr_matrix = df[nuemrical_cols].corr 
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            linewidths=.5, fmt=".2f")
sns.pairplot(numerical_df)

# QQ plot 
import scipy.stats as stats
stats.probplot(df['Age'], dist="norm", plot=plt)


# Missing Values: 

df.isnull().sum() 
df.isnull().sum(axis = 1) # row 
#remove cols based on threshold 
threshold_percent = 70; df = df.loc[:, df.isnull().mean() < threshold_percent/100] 
#remove rows based on threshold 
threshold_percent = 70; df = df.loc[df.isnull().mean(axis=1) < threshold_percent/100, :]

# MCAR - impute with mean, mode 
df['numerical_col'].fillna(df['numerical_col'].mean(), inplace=True)
df['categorical_col'].fillna(df['categorical_col'].mode()[0], inplace=True)

# MAR -
from sklearn.impute import KNNImputer

# Initialize KNNImputer
imputer = KNNImputer(n_neighbors=3)  # You can adjust the number of neighbors
# Impute the missing values
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)


# MNAR Subject Knowledge 

# Standardization 
from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Select numerical columns to be standardized
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Apply standardization only to numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Normalization 
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Select numerical columns to be normalized
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Apply normalization only to numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Log Transform 
#  Select numerical columns to be log-transformed
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].applymap(lambda x: np.log(x + 1e-10))

# SQRT Transform
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].applymap(np.aqrt)


# CATEGORICAL VARIABLES
target_encoded = pd.get_dummies(df['Target_Variable'], prefix='Target')
df = pd.concat([df.drop('Target_Variable', axis=1), target_encoded], axis=1)

from sklearn.preprocessing import LabelEncoder; label_encoder = LabelEncoder()
df['Target_Encoded'] = label_encoder.fit_transform(df['Target_Variable'])
df['Target_Decoded'] = label_encoder.inverse_transform(df['Target_Encoded'])

# EDA  - Categorical 
value_counts = df['Categorical_Column'].value_counts()
sns.countplot(data=df, x='Categorical_Column', palette="pastel") 


 # Create a cross-tabulation of the two categorical columns
cross_tab = pd.crosstab(index=df['Categorical_Column1'], columns=df['Categorical_Column2'])


# correlationn between nuymerical features and Target 
correlation_matrix = df.corr()
target_correlation = correlation_matrix['Target_Variable']


# correlation between numerical features and Target 
from scipy.stats import pointbiserialr
correlation, p_value = pointbiserialr(df['Numerical_Column'], df['Binary_Target'])


# adding label encoded features back to DF
# One-hot encode 'level' column
encoder = OneHotEncoder()
level_encoded = encoder.fit_transform(df[['level']]).toarray()
level_df = pd.DataFrame(level_encoded, columns=encoder.get_feature_names_out(['level']))
df = pd.concat([df, level_df], axis=1).drop(['level'], axis=1)
======================================================================
======================================================================


# dimensionality reduction -- PCA 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Standardize the dataset (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerical)

# Step 2: Initialize PCA and fit
pca = PCA(n_components=2)  # You can change the number of components
principal_components = pca.fit_transform(scaled_data)

# Step 3: Create a DataFrame with the principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# combine features after pre-processing
final_df = pd.concat([num_standardized, cat_encoded], axis=1)

# train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df, target_variable, test_size=0.2, random_state=42)

# Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training 
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions 
y_pred = model.predict(X_test)

# model evaluations 
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
===================================================
# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Making Predictions
dt_pred = dt_model.predict(X_test)
===================================================
# Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Making Predictions
rf_pred = rf_model.predict(X_test)
===================================================







===================================================








===================================================
# GRID search
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# For illustration purposes, creating a sample dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% testing

# Create Logistic Regression model
logreg = LogisticRegression(random_state=42)

# Train model
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)

# Model Evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(logreg, X, y, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Hyperparameter Tuning using GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid = GridSearchCV(logreg, param_grid, cv=5)
grid.fit(X_train, y_train)

# Print the best parameters found
print("Best Hyperparameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# You can then use grid.best_estimator_ as your model
final_model = grid.best_estimator_
final_predictions = final_model.predict(X_test)

# Final Model Evaluation
print("Final Accuracy:", metrics.accuracy_score(y_test, final_predictions))
print("Final Precision:", metrics.precision_score(y_test, final_predictions))
print("Final Recall:", metrics.recall_score(y_test, final_predictions))


===================================================
#Random Search CV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Hyperparameter grid
param_dist = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l1', 'l2'],
}

# Instantiating RandomizedSearchCV object
random_search = RandomizedSearchCV(
    logreg, param_distributions=param_dist,
    n_iter=100, cv=5, verbose=1, n_jobs=-1,
    random_state=42
)

# Fitting the model
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best Hyperparameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

# Use random_search.best_estimator_ as your model
final_model = random_search.best_estimator_
final_predictions = final_model.predict(X_test)

# Final Model Evaluation
print("Final Accuracy:", metrics.accuracy_score(y_test, final_predictions))
print("Final Precision:", metrics.precision_score(y_test, final_predictions))
print("Final Recall:", metrics.recall_score(y_test, final_predictions))








===================================================
NLP 

# Assuming you have a DataFrame named 'df' with a text column named 'text_column'
df['text_column'] = df['text_column'].str.lower()

# You might need to download the NLTK data by running nltk.download('punkt')
df['text_column'] = df['text_column'].apply(word_tokenize)

# stop words 
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

df['text_column'] = df['text_column'].apply(remove_stopwords)


# Lemmatization 
from nltk.stem import WordNetLemmatizer

# You might need to download the NLTK data by running nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return [lemmatizer.lemmatize(word) for word in text]

df['text_column'] = df['text_column'].apply(lemmatize_words)

===================================================================
# Regex Pattersn

import re

text = 'There are 2 apples, 5 oranges, and 3 bananas.'
result = re.sub(r'\d+', 'NUMBER', text)
print(result)  # Output: There are NUMBER apples, NUMBER oranges, and NUMBER bananas.

text = 'Hello, world! How are you today? :)'
result = re.sub(r'[^\w\s]', '', text)
print(result)  # Output: Hello world How are you today

text = 'Today’s date is 2023-09-15 and tomorrow is 2023-09-16.'
result = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', text)
print(result)  # Output: Today’s date is DATE and tomorrow is DATE.

text = 'I have a cat, but I like dogs too.'
result = re.sub(r'dogs|cat', 'animal', text)
print(result)  # Output: I have a animal, but I like animals too.


text = 'This    is a  text\t with   irregular   spacing.'
result = re.sub(r'\s+', ' ', text)
print(result)  # Output: This is a text with irregular spacing.


# lambda

result = df.apply(lambda x: x**2)
result = s.map({'cat': 'lion', 'dog': 'wolf'})
result = df.filter(items=['A', 'B'])
product = reduce(lambda x, y: x * y, df['A'])
iltered_df = df.filter(['A', 'B'])
# 2. Applying Function to Column 'A' (Using apply())
squared_series = filtered_df['A'].apply(lambda x: x**2)
# 3. Mapping Squared Values to Strings (Using map())
result_series = squared_series.map(lambda x: str(x))





================================================================================================================================
Regex

| Command | Description |
|---------|-------------|
| `.` | Matches any character except a newline. |
| `^` | Anchors the start of a line. |
| `$` | Anchors the end of a line. |
| `*` | Matches 0 or more repetitions of the preceding character. |
| `+` | Matches 1 or more repetitions of the preceding character. |
| `?` | Matches 0 or 1 occurrence of the preceding character. |
| `{m}` | Matches exactly m occurrences of the preceding character. |
| `{m,}` | Matches m or more occurrences of the preceding character. |
| `{m,n}` | Matches between m and n occurrences of the preceding character. |
| `\\` | Escapes a special character. |
| `[]` | Matches any single character within the brackets. |
| `|` | Acts as an OR operator. |
| `()` | Groups expressions as a single unit. |
|


serch
match 
findall 




