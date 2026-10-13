import requests

url = "http://127.0.0.1:5000/generate"

data = {
    "prompt": "What is artificial intelligence?"
}

response = requests.post(url, json=data)

print(response.json())
