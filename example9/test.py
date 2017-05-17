#!/usr/bin/env python

"""
Test script
"""

import csv
import classifier
import pprint


def main():
    result = classifier.predict([
        'Another warm day inland with increasing sunshine after early morning clouds and a few widely scattered showers',
        'Canceled a doctors appointment because it is too cold outside'
    ])
    pprint.pprint(result)


if __name__ == "__main__":
    main()
