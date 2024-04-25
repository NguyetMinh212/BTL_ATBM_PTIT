
from predict import Predictor
import pandas as pd
from csv import DictWriter
import csv 
list_of_urls = [
    {'url': 'http://www.google.com', 'assert_result': 0},
    {'url': 'https://stackoverflow.com/questions/12897446/userscript-to-wait-for-page-to-load-before-executing-code-techniques', 'assert_result': 0},
    {'url': 'https://www.youtube.com/watch?v=9bZkp7q19f0', 'assert_result': 0},
    {'url': 'https://www.facebook.com/', 'assert_result': 0},
    {'url': 'https://www.amazon.com/', 'assert_result': 0},
    {'url': 'https://www.reddit.com/', 'assert_result': 0},
    {'url': 'https://www.wikipedia.org/', 'assert_result': 0},
    {'url': 'https://www.twitter.com/', 'assert_result': 0},
    {'url': 'https://www.instagram.com/', 'assert_result': 0},
    {'url': 'https://www.linkedin.com/', 'assert_result': 0},
    {'url': 'https://www.netflix.com/', 'assert_result': 0},
    {'url': 'https://www.ebay.com/', 'assert_result': 0},
    {'url': 'https://www.apple.com/', 'assert_result': 0},
    {'url': 'https://www.microsoft.com/', 'assert_result': 0},
    {'url': 'https://www.paypal.com/', 'assert_result': 0},
    {'url': 'https://www.adobe.com/', 'assert_result': 0},
    {'url': 'https://www.dropbox.com/', 'assert_result': 0},
    {'url': 'https://www.spotify.com/', 'assert_result': 0},
    {'url': 'https://www.quora.com/', 'assert_result': 0},
    {'url': 'https://www.pinterest.com/', 'assert_result': 0},
    {'url': 'https://www.tumblr.com/', 'assert_result': 0},
    {'url': 'https://stackoverflow.com', 'assert_result': 0},
]

def test_url():
    predictor = Predictor()
    for url in list_of_urls:
        print(f"Testing URL: {url['url']}, Expected Result: {url['assert_result']}")
        result = predictor.predict_url(url['url'])
        print(f"Predicted Result: {result}")

def insert_data():
    field_names = ['url', 'label', 'url_len', 'url_has_login', 'url_has_client', 'url_has_server', 'url_has_admin', 'url_has_ip', 'url_isshorted', 'url_count_dot', 'url_count_https', 'url_count_http', 'url_count_perc', 'url_count_hyphen', 'url_count_www', 'url_count_atrate', 'url_2bentropy', 'url_3bentropy', 'url_hamming_00', 'url_hamming_01', 'url_hamming_11', 'url_hamming_10']
    list_of_data = generate_data()
    with open('train_dataset.csv', 'a') as f:
        dict_writer = DictWriter(f, fieldnames=field_names)
        for data in list_of_data:
            dict_writer.writerow(data)
            print("Data inserted " + str(data))
        f.close()


def generate_data():
    url_list = [
        'http://www.google.com',
        'https://stackoverflow.com/questions/12897446/userscript-to-wait-for-page-to-load-before-executing-code-techniques',
        'https://www.youtube.com/watch?v=9bZkp7q19f0',
        'https://www.facebook.com/',
        'https://www.amazon.com/',
        'https://www.reddit.com/',
        'https://www.wikipedia.org/',
        'https://www.twitter.com/',
        'https://www.instagram.com/',
        'https://www.linkedin.com/',
        'https://www.netflix.com/',
        'https://www.ebay.com/',
        'https://www.apple.com/',
        'https://www.microsoft.com/',
        'https://www.paypal.com/',
        'https://www.adobe.com/',
        'https://www.dropbox.com/',
        'https://www.spotify.com/',
        'https://www.quora.com/',
        'https://www.pinterest.com/',
        'https://www.tumblr.com/',
        'https://stackoverflow.com',
    ]
    predict = Predictor()
    list_of_data = []
    for url in url_list:
        feature = predict.extract_features(url)
        dict_url = {
            'url': url,
            'label': 0,
            'url_len': len(url),
            'url_has_login': feature['url_has_login'],
            'url_has_client': feature['url_has_client'],
            'url_has_server': feature['url_has_server'],
            'url_has_admin': feature['url_has_admin'],
            'url_has_ip': feature['url_has_ip'],
            'url_isshorted': feature['url_isshorted'],
            'url_count_dot': feature['url_count_dot'],
            'url_count_https': feature['url_count_https'],
            'url_count_http': feature['url_count_http'],
            'url_count_perc': feature['url_count_perc'],
            'url_count_hyphen': feature['url_count_hyphen'],
            'url_count_www': feature['url_count_www'],
            'url_count_atrate': feature['url_count_atrate']
        }
        list_of_data.append(dict_url)
    return list_of_data

def read_data():
    count = 0
    with open('test_dataset.csv', 'r') as textfile:
        for row in reversed(list(csv.reader(textfile))):
            print(row)
            count += 1
            if (count == 10):
                break

test_url()