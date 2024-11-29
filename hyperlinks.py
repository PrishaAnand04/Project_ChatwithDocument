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

def extract_hyperlinks(base_url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    links = soup.find_all('a', href=True)
    all_links = [urljoin(base_url, link['href']) for link in links[20:40]]
    return all_links

def extract_content_from_links(links):
    for i, link in enumerate(links):
        content = get_page_content(link)
        if content:
            print(f"\nContent from link {i + 1} ({link}):\n")
            print(content[:1000])  
        else:
            print(f"\nFailed to retrieve content from link {i + 1} ({link})")

url = "https://en.wikipedia.org/wiki/OpenAI"  
page_content = get_page_content(url)

if page_content:
    top_links = extract_hyperlinks(url, page_content)
    if top_links:
        print("First 20 hyperlinks on the page:")
        for i, link in enumerate(top_links):
            print(f"{i + 1}: {link}")
        extract_content_from_links(top_links)
    else:
        print("No hyperlinks found on the page.")
else:
    print("Failed to retrieve the webpage content.")
