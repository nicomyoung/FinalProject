import requests
from bs4 import BeautifulSoup
import time
import os

# Main page URL
URL = "https://www.npr.org/podcasts/510310/npr-politics-podcast"

# Base AJAX URL for subsequent pages
BASE_AJAX_URL = "https://www.npr.org/get/510310/render/partial/next?start="

# Ensure the directory exists
if not os.path.exists('C:\\Final Project'):
    os.makedirs('C:\\Final Project')

# Helper function to download audio
def download_audio(links):
    for link in links:
        time.sleep(1)
        audio_response = requests.get(link)
        filename = os.path.join('C:\\Final Project', f"podcast_{link.split('/')[-1].split('?')[0]}")
        with open(filename, 'wb') as file:
            file.write(audio_response.content)

# First, scrape the main page for audio links
response = requests.get(URL)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')
audio_links = [a['href'] for a in soup.select('li.audio-tool-download a')]
#download_audio(audio_links)  # Download main page audio immediately
print(f"Downloaded {len(audio_links)} audio links from main page.")

# Then, use the AJAX URL to get audio links from subsequent pages
#print("start: ", len(audio_links) + 1 )
start =  11
while True:
    ajax_url = BASE_AJAX_URL + str(start)
    response = requests.get(ajax_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    audio_links = [a['href'] for a in soup.select('li.audio-tool-download a')]

    if not audio_links:
        break

    download_audio(audio_links)  # Download each batch immediately
    print(f"Downloaded {len(audio_links)} audio links from AJAX page starting at {start}.")
    start += len(audio_links)  
    time.sleep(1)

print("All podcasts downloaded!")
