import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1. List all the Geico insurance URLs you want to ingest
URLS = [
    "https://www.geico.com/information/aboutinsurance/homeowners/",
    "https://www.geico.com/information/aboutinsurance/motorcycle/",
    "https://www.geico.com/information/aboutinsurance/atv/",
    "https://www.geico.com/information/aboutinsurance/umbrella/",
    "https://www.geico.com/information/aboutinsurance/renters/",
    "https://www.geico.com/information/aboutinsurance/condo/",
    "https://www.geico.com/information/aboutinsurance/co-op/",
    "https://www.geico.com/information/aboutinsurance/rv/",
    "https://www.geico.com/information/aboutinsurance/life/",
    "https://www.geico.com/information/aboutinsurance/boat/",
    "https://www.geico.com/information/aboutinsurance/watercraft/",
    "https://www.geico.com/information/aboutinsurance/flood/",
    "https://www.geico.com/information/aboutinsurance/mobilehome/",
    "https://www.geico.com/information/aboutinsurance/overseas/",
    "https://www.geico.com/information/aboutinsurance/travel/",
    "https://www.geico.com/information/aboutinsurance/commercial/",
    "https://www.geico.com/information/aboutinsurance/business/",
    "https://www.geico.com/information/aboutinsurance/id-theft/",
    "https://www.geico.com/information/aboutinsurance/snowmobile/",
    "https://www.geico.com/information/aboutinsurance/collectorcar/",
    "https://www.geico.com/information/aboutinsurance/mexico/",
    "https://www.geico.com/information/aboutinsurance/pet/",
    "https://www.geico.com/information/aboutinsurance/mobile-device-protection/",
    "https://www.geico.com/information/aboutinsurance/jewelry/",
    "https://www.geico.com/information/aboutinsurance/medical-malpractice/",
    "https://www.geico.com/information/aboutinsurance/earthquake/",
    "https://www.geico.com/information/aboutinsurance/event/"
]

def load_and_chunk():
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    for url in URLS:
        print(f"Fetching {url}")
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        # For each section heading, grab its text
        for header in soup.select("h2, h3"):
            title = header.get_text(strip=True)
            body = []
            for sib in header.find_next_siblings():
                if sib.name in ["h2", "h3"]:
                    break
                body.append(sib.get_text(strip=True))
            if not body:
                continue

            section_text = f"## {title}\n\n" + "\n\n".join(body)
            # Split into chunks and attach metadata
            for chunk in splitter.split_text(section_text):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"source": url, "section": title}
                ))

    return all_docs

if __name__ == "__main__":
    docs = load_and_chunk()
    print(f"Loaded {len(docs)} chunks from {len(URLS)} pages.")
