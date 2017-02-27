"""
Test 3

Introduces the use of more scikit libs to make preprocessing easier
"""
import csv
from sklearn.model_selection import KFold


print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)

# WIP...