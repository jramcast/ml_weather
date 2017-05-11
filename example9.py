from sklearn.externals import joblib

clf = joblib.load('models/k1.pkl')
result = clf.predict("Another warm day inland with increasing sunshine after early morning clouds and a few widely scattered showers")
print(result)