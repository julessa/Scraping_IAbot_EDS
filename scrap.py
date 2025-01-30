import requests
from bs4 import BeautifulSoup
import json
import re
import time
from collections import defaultdict

def get_soup(url, headers, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête {url} (tentative {attempt + 1}): {e}")
            time.sleep(2)  # Pause avant une nouvelle tentative
    return None

def extract_date(text):
    date_pattern = re.compile(r"\b(\d{1,2} [a-zéû]+ (\d{4}))\b", re.IGNORECASE)
    match = date_pattern.search(text)
    return match.groups() if match else (None, None)

def page_contains_valid_date(soup):
    """ Vérifie si la page contient au moins une date valide (1914-1945) """
    for element in soup.find_all():
        text = element.get_text(strip=True)
        _, year = extract_date(text)
        if year and (1914 <= int(year) <= 1945):
            return True
        if year and (0 <= int(year) <= 1913 or 1946 <= int(year) <= 2025):
            return False  # Exclure si on trouve une date hors plage
    return True  # Aucune date trouvée, on scrape quand même

def scrape_page(url, headers):
    time.sleep(2)  # Pause pour éviter d'être détecté comme un bot
    soup = get_soup(url, headers)
    if not soup or not page_contains_valid_date(soup):
        return []
    
    data = []
    for element in soup.find_all():
        text = element.get_text(strip=True)
        date_text, year = extract_date(text)
        if date_text and year and 1914 <= int(year) <= 1945:
            paragraph_text = ""
            next_sibling = element.find_next_sibling()
            while next_sibling:
                if next_sibling.name == "p":
                    paragraph_text += " " + next_sibling.get_text(strip=True)
                next_sibling = next_sibling.find_next_sibling()
            
            if paragraph_text:
                data.append({"date": date_text, "event": paragraph_text})
    
    return data

def scrape_site(base_url, headers, domain, max_pages=1000):
    soup = get_soup(base_url, headers)
    if not soup:
        return []
    
    all_data = scrape_page(base_url, headers)
    visited_urls = set()
    to_visit = [base_url]
    page_count = 0

    while to_visit and page_count < max_pages:
        url = to_visit.pop()
        if url in visited_urls:
            continue
        visited_urls.add(url)
        page_count += 1

        print(f"Scraping ({page_count}/{max_pages}): {url}")
        soup = get_soup(url, headers)
        if not soup or not page_contains_valid_date(soup):
            continue

        all_data.extend(scrape_page(url, headers))

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if (href.startswith("/") or domain in href) and "http" not in href:
                full_url = f"https://{domain}{href}" if href.startswith("/") else href
                if full_url not in visited_urls:
                    to_visit.append(full_url)
    
    return all_data

def merge_data(*datasets):
    merged_data = []
    date_count = defaultdict(int)

    for dataset in datasets:
        for entry in dataset:
            date_count[entry["date"]] += 1
            merged_data.append({
                "date": entry["date"],
                "event": entry["event"],
                "duplicate": date_count[entry["date"]] > 1
            })

    return merged_data

if __name__ == "__main__":
    headers = {"User-Agent": "Mozilla/5.0"}
    herodote_data = scrape_site("https://www.herodote.net/La_Grande_Guerre_ou_Premiere_Guerre_mondiale-synthese-60.php", headers, "herodote.net")
    histoire_pour_tous_data = scrape_site("https://www.histoire-pour-tous.fr/histoire-de-france/212-dune-guerre-mondiale-a-lautre.html", headers, "histoire-pour-tous.fr")

    all_data = merge_data(herodote_data, histoire_pour_tous_data)

    with open("combined_data.json", "w", encoding="utf-8") as json_file:
        json.dump(all_data, json_file, ensure_ascii=False, indent=4)

    print(f"Export JSON terminé: {len(all_data)} entrées enregistrées dans combined_data.json")
