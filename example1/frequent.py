import csv
import string
from pprint import pprint
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import TweetTokenizer


# Prepare stopwords
punctuation = list(string.punctuation)
stopwords_list = stopwords.words('english') + punctuation \
                    + ['rt', 'via', '…', '...', '️', 'ヽ', '、', '｀' ]


def get_most_frequent_terms(tweets, how_many):
    tokenizer = TweetTokenizer(preserve_case=False)
    counter = Counter()

    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet)
        filtered_tokens = [token for token in tokens if token not in stopwords_list]
        counter.update(filtered_tokens)

    return counter.most_common(how_many) 



