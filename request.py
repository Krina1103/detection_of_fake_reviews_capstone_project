import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'review': 'super deal for this mattress.  The mattress is comfortable and its also a very strong pill.'})

print(r.json())