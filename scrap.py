
import requests
from bs4 import BeautifulSoup
import json
import re

def get_soup(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête {url}: {e}")
        return None

def extract_date(text):
    # Regex pour capturer des dates sous forme "12 janvier 1918" ou variantes
    date_pattern = re.compile(r"\b(\d{1,2} [a-zéû]+ (\d{4}))\b", re.IGNORECASE)
    match = date_pattern.search(text)
    return match.groups() if match else (None, None)

def scrape_page(url, headers):
    soup = get_soup(url, headers)
    if not soup:
        return []
    
    data = []
    for element in soup.find_all():  # Parcourir toutes les balises
        text = element.get_text(strip=True)
        date_text, year = extract_date(text)
        if date_text and year and 1914 <= int(year) <= 1945:
            next_sibling = element.find_next_sibling()
            
            while next_sibling and next_sibling.name not in ["p", "br"]:
                next_sibling = next_sibling.find_next_sibling()
            
            if next_sibling and next_sibling.name == "p":
                event_text = next_sibling.get_text(strip=True)
                data.append({"date": date_text, "event": event_text})
    
    return data

def scrape_herodote():
    base_url = "https://www.herodote.net/La_Grande_Guerre_ou_Premiere_Guerre_mondiale-synthese-60.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    soup = get_soup(base_url, headers)
    if not soup:
        return []
    
    all_data = scrape_page(base_url, headers)
    
    visited_urls = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/") or "herodote.net" in href:  # Vérifier que le lien reste sur Herodote.net
            full_url = href if "herodote.net" in href else "https://www.herodote.net" + href
            if full_url not in visited_urls:
                visited_urls.add(full_url)
                all_data.extend(scrape_page(full_url, headers))
    
    return all_data

if __name__ == "__main__":
    data = scrape_herodote()
    if data:
        with open("herodote_data.json", "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Export JSON terminé: {len(data)} entrées enregistrées dans herodote_data.json")
    else:
        print("Aucune donnée trouvée.")