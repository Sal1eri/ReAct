import requests

url = "https://en.wikipedia.org/w/index.php?search=Scott+Derrickson"
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

r = requests.get(url, proxies=proxies, timeout=20)
print(r.status_code)