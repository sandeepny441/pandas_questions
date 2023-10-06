# Numerical EDA:

sns.histplot(df[col])
sns.kdeplot(df[col])
df[col].kurtosis()
df[col].skew()
sns.boxplot(df[col])
q1 = df[col].quantile(25)
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








