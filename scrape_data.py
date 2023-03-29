import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

url_base = "https://holtsmithsonfoundation.org"
url_endings_holt = ["hydras-head-0", "catch-basin", "ventilation-series",
                    "dark-star-park-0", "pipeline-0", "sun-tunnels-0", "pine-barrens-0", "holes-light-0"]

data_holt = []

for ending in url_endings_holt:
    url = f"{url_base}/{ending}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_title = soup.find("h1").text.strip()
    paragraphs = []
    for p in soup.find_all('p'):
        if len(p.text.split()) >= 50:
            paragraphs.append(p.text.strip())
    for paragraph in paragraphs:
        data_holt.append(
            {'Article Title': article_title, 'Paragraph': paragraph})

df_holt = pd.DataFrame(data_holt)

# Write dataframe to CSV file in data directory
df_holt.to_csv(os.path.join("data", "holt_data.csv"), index=False)


url_endings_smithson = ["spiral-jetty-1", "interpolation-enantiomorphic-chambers", "entropy-and-new-monuments",                        "tour-monuments-passaic-new-jersey", "provisional-theory-nonsites"]

data_smithson = []

for ending in url_endings_smithson:
    url = f"{url_base}/{ending}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_title = soup.find("h1").text.strip()
    paragraphs = []
    for p in soup.find_all('p'):
        if len(p.text.split()) >= 50:
            paragraphs.append(p.text.strip())
    for paragraph in paragraphs:
        data_smithson.append(
            {'Article Title': article_title, 'Paragraph': paragraph})

df_smithson = pd.DataFrame(data_smithson)
df_smithson.to_csv(os.path.join("data", "smithson_data.csv"), index=False)
