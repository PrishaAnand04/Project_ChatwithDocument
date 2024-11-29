import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_first_20_hyperlinks(base_url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    links = soup.find_all('a', href=True)
    all_links = [urljoin(base_url, link['href']) for link in links[21:25]] #on wiki pedia start with 22 no. of hplink
    return all_links

url = "https://en.wikipedia.org/wiki/OpenAI"  # Replace with the target URL
page_content = get_page_content(url)

if page_content:
    first_20_links = extract_first_20_hyperlinks(url, page_content)
    if first_20_links:
        print("\nFirst 20 hyperlinks on the page:")
        for i, link in enumerate(first_20_links):
            print(f"{i + 1}: {link}")
    else:
        print("No hyperlinks found on the page.")
else:
    print("Failed to retrieve the webpage content.")
