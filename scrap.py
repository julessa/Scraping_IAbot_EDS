import requests
from bs4 import BeautifulSoup

def scrape_herodote():
    url = "https://www.herodote.net/La_Grande_Guerre_ou_Premiere_Guerre_mondiale-synthese-60.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extraire les titres (h1, h2, h3, etc.)
    titles = [title.get_text(strip=True) for title in soup.find_all(['h1', 'h2', 'h3'])]
    
    # Extraire le texte des paragraphes
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    
    # Regrouper les données
    scraped_data = {
        "titles": titles,
        "paragraphs": paragraphs
    }
    
    return scraped_data

if __name__ == "__main__":
    data = scrape_herodote()
    if data:
        print("\nTitres extraits:")
        for title in data["titles"]:
            print(f"- {title}")
        
        print("\nParagraphes extraits:")
        for para in data["paragraphs"][:5]:  # Afficher seulement les 5 premiers pour l'aperçu
            print(f"- {para[:150]}...")