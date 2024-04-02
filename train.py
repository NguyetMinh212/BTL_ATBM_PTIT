import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import math
from collections import Counter


train_data = pd.read_csv('test_dataset.csv')
train_data.head()
test_data = pd.read_csv('test_dataset.csv')
test_data.head()
# Select non-string columns for x_train
x_train = train_data.iloc[:10000, 3:].select_dtypes(exclude=['object'])

# Select column 1 for y_train
y_train = train_data.iloc[:10000, 1]

# Select non-string columns for x_test
x_test = test_data.iloc[:10000, 3:].select_dtypes(exclude=['object'])

# Select column 1 for y_test
y_test = test_data.iloc[:10000, 1]

# Initialize Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=50, random_state=0)

# Calculate feature importance scores
random_forest.fit(x_train, y_train)
feature_scores = pd.Series(random_forest.feature_importances_,
                           index=x_train.columns).sort_values(ascending=False)

# Select the most important features
threshold = 0.03
selected_features = feature_scores[feature_scores >= threshold].index.tolist()

# Filter the training and test data based on selected features
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

# Retrain the model using selected features
random_forest.fit(x_train_selected, y_train)

# Make predictions
y_pred = random_forest.predict(x_test_selected)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision: ", precision)

recall = recall_score(y_test, y_pred)
print("Recall: ", recall)

f1 = f1_score(y_test, y_pred)
print("F-1 score: ", f1)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(conf_matrix)

# Display selected features and their importance scores
print("Selected Features and Their Importance Scores:")


def calculate_entropy(url):
    # Count the occurrences of each character in the URL
    counter = Counter(url)

    # Calculate the probabilities
    probabilities = [count / len(url) for count in counter.values()]

    # Calculate and return the entropy
    return -sum(p * math.log2(p) for p in probabilities)


# CODE TEST: PLEASE MODIFY, THANKS
def extract_features(url):
    # Initialize a dictionary to hold the features
    features = {}

    # Extract features from the URL
    features['url_len'] = len(url)
    # [Index(['url_len', 'url_entropy', 'path_count_no_of_dir', 'path_len',\n       'url_count_letter', 'path_count_lower', 'subdomain_len',\n       'pdomain_min_distance', 'url_2bentropy', 'url_hamming_1',\n       'url_3bentropy', 'url_nunique_chars_ratio', 'url_hamming_00',\n       'url_count_digit', 'url_hamming_01', 'url_hamming_11',\n       'url_hamming_10']

    features['url_entropy'] = calculate_entropy(url)

    # count of number of directories in the URL
    features['path_count_no_of_dir'] = url.count('/')

    # count of number of characters in the URL
    features['path_len'] = len(url)

    # count of number of letters in the URL
    features['url_count_letter'] = sum(c.isalpha() for c in url)

    # count of number of lowercase letters in the URL
    features['path_count_lower'] = sum(c.islower() for c in url)

    # length of the subdomain
    features['subdomain_len'] = len(url.split('.')[0])

    # minimum distance of the domain from the public suffix list
    features['pdomain_min_distance'] = 1

    # 2-gram entropy of the URL
    features['url_2bentropy'] = 1

    # hamming distance of the URL with the most common 1-gram
    features['url_hamming_1'] = 1

    # 3-gram entropy of the URL
    features['url_3bentropy'] = 1

    # ratio of the number of unique characters to the length of the URL
    features['url_nunique_chars_ratio'] = len(set(url))/len(url)

    # hamming distance of the URL with the most common 00-gram
    features['url_hamming_00'] = 1

    # count of number of digits in the URL
    features['url_count_digit'] = sum(c.isdigit() for c in url)

    # hamming distance of the URL with the most common 01-gram
    features['url_hamming_01'] = 1

    # hamming distance of the URL with the most common 11-gram
    features['url_hamming_11'] = 1

    # hamming distance of the URL with the most common 10-gram
    features['url_hamming_10'] = 1

    return pd.DataFrame([features])


# Function to predict a url
def predict_url(url):
    # remove https
    url = url.replace('https://', '')
    url = url.replace('http://', '')

    # remove www
    url = url.replace('www.', '')

    # remove trailing slash
    if url[-1] == '/':
        url = url[:-1]

    # predict
    x = extract_features(url)
    y = random_forest.predict(x)
    return y[0]


# Test the function
url_list = [
    "https://www.google.com",
    "https://w...content-available-to-author-only...k.com",
    "https://w...content-available-to-author-only...e.com",
    "https://w...content-available-to-author-only...r.com",
    "https://w...content-available-to-author-only...n.com",
    "https://w...content-available-to-author-only...m.com",
    "https://w...content-available-to-author-only...t.com",
    "https://w...content-available-to-author-only...r.com",
    "https://w...content-available-to-author-only...t.com",
    "https://w...content-available-to-author-only...s.com",
    "quochung.cyou",
    "https://d...content-available-to-author-only...m.vn/unikey-6799"
]

malicious_url_list = [
    "ne01u59l.firebaseapp.com",
    "royal-reschedule-parcel.com",
    "pge152932id0724478restricted.co.vu/invalid.php",
    "tinyurl.com/joncarlsen",
    "aguasandinas.online-corpweb.com/c/p/tc/21337/3ca52a26-3f88-40bf-80a6-e4b9b5ea9d0c-104ac7de-709d-4b52"
]

for url in url_list:
    prediction = predict_url(url)
    print(url, prediction)

for url in malicious_url_list:
    prediction = predict_url(url)
    print(url, prediction)
