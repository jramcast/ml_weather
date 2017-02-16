import csv
with open('data/train.csv', newline='') as csvfile:
	datareader = csv.DictReader(csvfile)
	for row in datareader:
		print(row['id'], row['s1'])