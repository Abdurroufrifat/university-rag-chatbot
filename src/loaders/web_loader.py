import requests
from bs4 import BeautifulSoup

def load_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)

        if text:
            return [{
                "text": text,
                "metadata": {
                    "source": url,
                    "page": "web",
                    "type": "web"
                }
            }]
        return []

    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []