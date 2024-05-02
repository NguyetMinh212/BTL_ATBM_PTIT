# Prepare data
train_data = pd.read_csv('train_dataset.csv')

#print first row with column name and data
print("Loaded csv file " + str(train_data.shape[0]) + " rows and " + str(train_data.shape[1]) + " columns")


test_data = pd.read_csv('test_dataset.csv')
print("Loaded dataset")
print("Loaded test dataset " + str(test_data.shape[0]) + " rows and " + str(test_data.shape[1]) + " columns")