import requests

ACCESS_TOKEN = "sKuuQOA2d1lrkBY4PP0a15Qd909qD4UYvRDSfItRGWt6IM55wjVUgz6kIQuf"
record_id = "14247083"




response = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})

print("Request URL:", response.request.url)
print("Request Method:", response.request.method)
print("Request Headers:", response.request.headers)
print("Request Body:", response.request.body)

download_urls = [f['links']['self'] for f in response.json()['files']]
filenames = [f['key'] for f in response.json()['files']]

print(response.status_code)
print(download_urls)



for filename, url in zip(filenames, download_urls):
    print("Downloading:", filename)
    r = requests.get(url, params={'access_token': ACCESS_TOKEN})
    with open(filename, 'wb') as f:
        f.write(r.content)
