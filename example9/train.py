#!/usr/bin/env python

"""
Train script
"""

import csv
import classifier


def main():
    print('Reading CSV...')
    csvfile = open('../data/train.csv', newline='')
    datareader = csv.DictReader(csvfile)
    data = list(datareader)
    print('Training...')
    classifier.train(data)


if __name__ == "__main__":
    main()
