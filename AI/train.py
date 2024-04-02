import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from sklearn.model_selection import GridSearchCV

# Prepare data
train_data = pd.read_csv('train_dataset.csv')
train_data.head()
test_data = pd.read_csv('test_dataset.csv')
test_data.head()
print("Loaded dataset")
print(train_data.shape)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}



# Select non-string columns for x_train
x_train = train_data.iloc[:10000, 3:].select_dtypes(exclude=['object']).drop(columns=['url_2bentropy', 'url_3bentropy', 'url_hamming_00', 'url_hamming_01', 'url_hamming_11', 'url_hamming_10'])
# Select column 1 for y_train
y_train = train_data.iloc[:10000, 1]
# Select non-string columns for x_test
x_test = test_data.iloc[:10000, 3:].select_dtypes(exclude=['object']).drop(columns=['url_2bentropy', 'url_3bentropy', 'url_hamming_00', 'url_hamming_01', 'url_hamming_11', 'url_hamming_10'])
# Select column 1 for y_test
y_test = test_data.iloc[:10000, 1]


#Print out all feature name
feature_names = x_train.columns.tolist()


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


# Initialize the grid search
# GridSearchCV is a utility function from sklearn that performs a grid search over specified parameter values for an estimator.
# It exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter.
# In this case, the estimator is a RandomForestClassifier and the parameters to search over are defined in the param_grid.
grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, scoring='f1')
# Fit the grid search
grid_search.fit(x_train_selected, y_train)
# Get the best parameters
best_params = grid_search.best_params_

# Train a new model with the best parameters
# A new RandomForestClassifier is initialized with the best parameters found by the grid search.
# The model is then fit with the training data.
best_model = RandomForestClassifier(**best_params, random_state=0)
best_model.fit(x_train_selected, y_train)


# Make predictions with the new model
# The predict method is used to predict the class labels for the provided data.
# The data passed to predict are the test data.
y_pred = best_model.predict(x_test_selected)

# Evaluate the new model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision: ", precision)
recall = recall_score(y_test, y_pred)
print("Recall: ", recall)
f1 = f1_score(y_test, y_pred)
print("F-1 score: ", f1)



# Save the pickled model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


print("Model saved as random_forest_model.pkl")


#Save the selected features to a file
with open('selected_features.pkl', 'wb') as file:
    pickle.dump(selected_features, file)

print("Selected features saved as selected_features.pkl")