import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def convert_to_numeric(df):
    """
    Converts all columns in a DataFrame to numeric values using LabelEncoder.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to convert.

    Returns:
    - pandas.DataFrame: The DataFrame with all columns converted to numeric values.

    This function iterates over all columns in the DataFrame and applies LabelEncoder.fit_transform() to each column. The converted values are then assigned back to the DataFrame. The function prints a message indicating the number of columns being converted.

    Example usage:
    ```
    df = convert_to_numeric(df)
    ```
    """
    print("[Preprocessing] Converting " + str(df.shape[1]) + " columns to numeric")
    for column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df



def load_data(file_name, pickle_file):
    """
    Load data from a CSV file or a pickled file.
    Pickled file is a format for faster reading, will be saved first time read as CSV for faster process later

    Parameters:
    - file_name (str): The path to the CSV file if a pickled version does not exist.
    - pickle_file (str): The path to the pickled file.

    Returns:
    - pandas.DataFrame: The loaded data from the CSV file or the pickled file.

    This function checks if a pickled version of the file already exists. If it does, it loads the data from the pickled file using `pd.read_pickle()`. If the pickled file does not exist, it loads the data from the CSV file using `pd.read_csv()` and then pickles it for future use. The loaded data is returned as a pandas DataFrame.

    The function prints the file path being loaded and the shape of the loaded dataset. It also prints the file path where the data is being pickled.

    Example usage:
    ```
    data = load_data('data.csv', 'data.pkl')
    ```
    """
    # Check if a pickled version of the file already exists
    if os.path.exists(pickle_file):
        print(f"Loading data from {pickle_file}")
        return pd.read_pickle(pickle_file)
    else:
        # If not, load from CSV and then pickle it for next time
        print(f"Loading data from {file_name}")
        data = pd.read_csv(file_name)
        print(f"[Load Data] Loaded dataset {data.shape[0]} rows and {data.shape[1]} columns")
        
        # 'wb' is write binary
        with open(pickle_file, 'wb') as file:
            pd.to_pickle(data, file)
        print(f"[Load Data] Data pickled at {pickle_file}")
        return data


# Pick a binary format file name
pickle_file_train = 'train_dataset.pkl'
pickle_file_test = 'test_dataset.pkl'
# Prepare data
train_data = load_data('train_dataset.csv', pickle_file_train)
test_data = load_data('test_dataset.csv', pickle_file_test)







# TRAIN DATA

# Select non-string columns for x_train
columns_to_drop = ['url_2bentropy', 'url_3bentropy', 'url_hamming_00', 'url_hamming_01', 'url_hamming_11', 'url_hamming_10']
checked_columns_to_drop = [col for col in columns_to_drop if col in train_data.columns]
train_data = convert_to_numeric(train_data)
# Select column 1 for y_train, non-string columns
x_train = train_data.iloc[:, 3:].select_dtypes(exclude=['object']).drop(columns=checked_columns_to_drop)
y_train = train_data.iloc[:, 1]



# TEST DATA

checked_columns_to_drop = [col for col in columns_to_drop if col in test_data.columns]
# Select column 1 for y_test, non-string columns
x_test = test_data.iloc[:, 3:].select_dtypes(exclude=['object']).drop(columns=checked_columns_to_drop)
y_test = test_data.iloc[:, 1]
test_data = convert_to_numeric(test_data)


feature_names = x_train.columns.tolist()
print("[Training] Random forest model training")


# Initialize Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=50, random_state=0)
# Calculate feature importance scores
random_forest.fit(x_train, y_train)
feature_scores = pd.Series(random_forest.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# Select the most important features
threshold = 0.03
selected_features = feature_scores[feature_scores >= threshold].index.tolist()
# Filter the training and test data based on selected features
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]




# Make predictions with the new model
# The predict method is used to predict the class labels for the provided data.
# The data passed to predict are the test data.
y_pred = random_forest.predict(x_test_selected)

# Evaluate the new model
accuracy = accuracy_score(y_test, y_pred)
print("[Test Result] Accuracy: ", accuracy)
precision = precision_score(y_test, y_pred)
print("[Test Result] Precision: ", precision)
recall = recall_score(y_test, y_pred)
print("[Test Result] Recall: ", recall)
f1 = f1_score(y_test, y_pred)
print("[Test Result] F-1 score: ", f1)



# Save the pickled model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest, file)
    print("[Persist] Model saved as random_forest_model.pkl")


#Save the selected features to a file
with open('selected_features.pkl', 'wb') as file:
    pickle.dump(selected_features, file)
    print("[Persist] Selected features saved as selected_features.pkl")