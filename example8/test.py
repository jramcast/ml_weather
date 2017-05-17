from sklearn.externals import joblib
from preprocessing import *


clf = joblib.load('models/k1.pkl')
result = clf.predict([
    "Another warm day inland with increasing sunshine after early morning clouds and a few widely scattered showers"
])
print(result)