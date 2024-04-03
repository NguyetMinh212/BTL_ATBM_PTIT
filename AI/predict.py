import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
from collections import Counter
import pickle

class Predictor:
    def __init__(self):
        # Load the pickled model
        with open('random_forest_model.pkl', 'rb') as file:
            self.loaded_model = pickle.load(file)

        #Open selected_features.pkl file
        with open('selected_features.pkl', 'rb') as file:
            self.selected_features = pickle.load(file)

    # All other functions are now methods of the Predictor class.
    # Replace all instances of 'loaded_model' and 'selected_features' with 'self.loaded_model' and 'self.selected_features'

    def calculate_entropy(self, url):
        """
        This function calculates the entropy of a given URL.
        Entropy is a measure of randomness or unpredictability in the data.
        In this case, it's calculated as the sum of the probability of each character
        in the URL times the log base 2 of that probability, all negated.

        Parameters:
        url (str): The URL for which the entropy is to be calculated.

        Returns:
        float: The calculated entropy.
        """
        counter = Counter(url)
        probabilities = [count / len(url) for count in counter.values()]
        return -sum(p * math.log2(p) for p in probabilities)

    def calculateHammingDistance(self, url):
        """
        This function calculates the Hamming distance for a given URL.
        The Hamming distance is defined as the number of positions at which
        the corresponding symbols are different. In this case, it's the number
        of times a character in the URL is not equal to the next character.

        Parameters:
        url (str): The URL for which the Hamming distance is to be calculated.

        Returns:
        int: The calculated Hamming distance.
        """
        hamming_distance = 0
        for i in range(len(url)-1):
            if url[i] != url[i+1]:
                hamming_distance += 1
        return hamming_distance

    def extract_features(self, url):
        features = {}
        features['url_has_login'] = 'login' in url
        features['url_has_client'] = 'client' in url
        features['url_has_server'] = 'server' in url
        features['url_has_admin'] = 'admin' in url
        features['url_has_ip'] = 'ip' in url
        features['url_isshorted'] = 'bit.ly' in url or 'tinyurl' in url or 'shorturl' in url or 'shorte' in url or 'short.ly' in url
        features['url_count_dot'] = url.count('.')
        features['url_count_https'] = url.count('https')
        features['url_count_http'] = url.count('http')
        features['url_count_perc'] = url.count('%')
        features['url_count_hyphen'] = url.count('-')
        features['url_count_www'] = url.count('www')
        features['url_count_atrate'] = url.count('@')
        features['url_count_hash'] = url.count('#')
        features['url_count_semicolon'] = url.count(';')
        features['url_count_underscore'] = url.count('_')
        features['url_count_ques'] = url.count('?')
        features['url_count_equal'] = url.count('=')
        features['url_count_amp'] = url.count('&')
        features['url_count_sensitive_financial_words'] = sum(word in url for word in ['bank', 'paypal', 'account', 'credit', 'card', 'money', 'transfer', 'payment', 'login', 'signin', 'secure', 'verify', 'update', 'account', 'password', 'ssn', 'social', 'security', 'number', 'identity', 'theft', 'fraud', 'phishing', 'scam', 'hack', 'hijack', 'malware', 'virus', 'trojan', 'keylogger', 'ransomware', 'phish', 'spoof', 'spoofing', 'identity', 'theft', 'fraud', 'scam', 'hack', 'hijack', 'malware', 'virus', 'trojan', 'keylogger', 'ransomware', 'phish', 'spoof', 'spoofing'])
        features['url_count_sensitive_words'] = sum(word in url for word in ['login', 'signin', 'secure', 'verify', 'update', 'account', 'password', 'ssn', 'social', 'security', 'number', 'identity', 'theft', 'fraud', 'phishing', 'scam', 'hack', 'hijack', 'malware', 'virus', 'trojan', 'keylogger', 'ransomware', 'phish', 'spoof', 'spoofing', 'identity', 'theft', 'fraud', 'scam', 'hack', 'hijack', 'malware', 'virus', 'trojan', 'keylogger', 'ransomware', 'phish', 'spoof', 'spoofing'])
        features['path_has_any_sensitive_words'] = sum(word in url for word in ['login', 'signin', 'secure', 'verify', 'update', 'account', 'password', 'ssn', 'social', 'security', 'number', 'identity', 'theft', 'fraud', 'phishing', 'scam', 'hack', 'hijack', 'malware', 'virus', 'trojan', 'keylogger', 'ransomware', 'phish', 'spoof', 'spoofing', 'identity', 'theft', 'fraud', 'scam', 'hack', 'hijack', 'malware', 'virus', 'trojan', 'keylogger', 'ransomware', 'phish', 'spoof', 'spoofing'])
        features['path_has_singlechardir'] = sum(len(dir) == 1 for dir in url.split('/')[3:])
        features['path_has_upperdir'] = sum(dir.isupper() for dir in url.split('/')[3:])
        features['query_len'] = len(url.split('?')[-1])
        features['query_count_components'] = len(url.split('?')[-1].split('&'))
        features['pdomain_count_hyphen'] = url.split('.')[-2].count('-')
        features['pdomain_count_atrate'] = url.split('.')[-2].count('@')
        features['pdomain_count_non_alphanum'] = sum(not c.isalnum() for c in url.split('.')[-2])
        features['pdomain_count_digit'] = sum(c.isdigit() for c in url.split('.')[-2])
        features['tld_is_sus'] = url.split('.')[-1] in ['xyz', 'top', 'club', 'online', 'site', 'website', 'space', 'tech', 'info', 'bid', 'win', 'vip', 'guru', 'work', 'party', 'click', 'link', 'help', 'support', 'download', 'racing', 'stream', 'live', 'video', 'gq', 'kim', 'loan']
        features['path_count_no_of_embed'] = url.count('embed')
        features['path_count_zero'] = url.count('0')
        features['path_count_pertwent'] = url.count('%20')
        features['path_count_lower'] = sum(c.islower() for c in url)
        features['path_count_upper'] = sum(c.isupper() for c in url)
        features['path_count_nonascii'] = sum(not c.isascii() for c in url)
        features['path_has_singlechardir'] = sum(len(dir) == 1 for dir in url.split('/')[3:])
        features['path_has_upperdir'] = sum(dir.isupper() for dir in url.split('/')[3:])
        features['url_entropy'] = self.calculate_entropy(url)
        features['url_hamming_1'] = self.calculateHammingDistance(url)
        features['url_len'] = len(url)
        features['path_count_no_of_dir'] = len(url.split('/')) - 3
        features['url_count_letter'] = sum(c.isalpha() for c in url)
        features['subdomain_len'] = len(url.split('.')[0])
        features['pdomain_min_distance'] = min(url.split('.')[-2].find(c) for c in ['-', '@', '.', '_'])
        features['url_nunique_chars_ratio'] = len(set(url)) / len(url)
        features['path_len'] = len(url.split('?')[0].split('/')[-1])
        features['url_count_digit'] = sum(c.isdigit() for c in url)
        features['tld_len'] = len(url.split('.')[-1])
        return pd.DataFrame([features])

    def predict_url(self, url):
        x_predict = self.extract_features(url)
        x_predict = x_predict[self.selected_features]
        y_predict = self.loaded_model.predict(x_predict)
        return y_predict[0]
