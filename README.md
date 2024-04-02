
# ATBM-2024 (Malicious URL Detection System)

Malicious URL Detection System is a project that aims to detect malicious URLs using Random Forest classifier. 

## Features

- Detect malicious URLs
- Detect automatically through browser extension
- Standalone website


## Requirements

- Python 3.8
- Python Lib
  - numpy
  - pandas
  - scikit-learn
  - seaborn
  - scikit-learn


## Installation

AI model is trained using Random Forest classifier. The model is saved in the `model` directory. 

### 3.1 Training the model
```bash
  - Using dataset from here https://www.kaggle.com/datasets/pilarpieiro/tabular-dataset-ready-for-malicious-url-detection/data?select=train_dataset.csv
  - Run the `train.py` file to train the model
  - Trained model and extracted features are saved in the `model` directory
```

### 3.2 Run prediction
```bash
  - Run the `predict.py` file to predict the URL
```


## License

[MIT](https://choosealicense.com/licenses/mit/)

