import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import openpyxl

BASE_URL = "https://catalog.unh.edu"
SITEMAP_FILE = "graduate_catalog.xlsx"
OUTPUT_FILE = "unh_catalog.json"

# helper functions
def clean_text(tag):
    return tag.get_text(" ", strip=True)

def heading_level(name):
    if not name:
        return 0
    name = name.lower()
    if name in ("h2", "h3", "h4"):
        return int(name[1])
    return 0

def push_section(level, section_title, stack):
    while stack and stack[-1][0] >= level:
        stack.pop()
    parent = stack[-1][1]
    new_node = {"type": "section", "title": section_title, "content": []}
    parent["content"].append(new_node)
    stack.append((level, new_node))

def add_paragraph(tag, stack, URL):
    _, current = stack[-1]
    item = {"type": "text", "text": clean_text(tag)}
    links = [
        {"label": clean_text(a), "url": urljoin(URL, a["href"].strip())}
        for a in tag.find_all("a", href=True)
    ]
    if links:
        item["links"] = links
    current["content"].append(item)

def add_list(tag, stack, URL):
    _, current = stack[-1]
    items = [clean_text(li) for li in tag.find_all("li", recursive=False)]
    item = {"type": "list", "items": items}
    links = [
        {"label": clean_text(a), "url": urljoin(URL, a["href"].strip())}
        for a in tag.find_all("a", href=True)
    ]
    if links:
        item["links"] = links
    current["content"].append(item)

def normalize(section_node):
    title = section_node["title"]
    text_blocks, list_blocks, link_bucket, children = [], [], [], []
    for item in section_node["content"]:
        if item["type"] == "text":
            text_blocks.append(item["text"])
            if "links" in item:
                link_bucket.extend(item["links"])
        elif item["type"] == "list":
            list_blocks.append(item["items"])
            if "links" in item:
                link_bucket.extend(item["links"])
        elif item["type"] == "section":
            children.append(normalize(item))
    out = {"title": title}
    if text_blocks:
        out["text"] = text_blocks
    if list_blocks:
        out["lists"] = list_blocks
    if link_bucket:
        out["links"] = link_bucket
    if children:
        out["subsections"] = children
    return out

# load the sitemap
wb = openpyxl.load_workbook(SITEMAP_FILE)
sheet = wb.active
rows = list(sheet.iter_rows(min_row=2, values_only=True))  # skip header

all_pages_data = []

for row in rows:
    # build full hierarchy from all non-empty columns except last
    titles = [str(c).strip() for c in row[:-1] if c]
    relative_url = str(row[-1]).strip()
    if not relative_url:
        continue

    full_title = " > ".join(titles)
    URL = urljoin(BASE_URL, relative_url)
    print(f"[info] fetching page: {URL} ({full_title})")

    try:
        resp = requests.get(URL, timeout=30)
        html = resp.text
    except Exception as e:
        print(f"[error] could not download {URL}: {e}")
        continue

    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup.find("article") or soup.find("div", id="content") or soup
    flow = main.select("h2, h3, h4, p, ul, ol")

    root = {"type": "root", "content": []}
    stack = [(1, root)]
    toc = []

    for el in flow:
        lvl = heading_level(el.name)
        if lvl:
            heading_title = clean_text(el)
            if lvl == 2:
                toc.append(heading_title)
            push_section(lvl, heading_title, stack)
        else:
            if el.name == "p":
                add_paragraph(el, stack, URL)
            elif el.name in ("ul", "ol"):
                add_list(el, stack, URL)

    sections = [normalize(n) for n in root["content"] if n.get("type") == "section"]

    page_data = {
        "page_title": full_title or clean_text(soup.find("h1")) or URL,
        "url": URL,
        "toc": toc,
        "sections": sections
    }

    all_pages_data.append(page_data)

# write all pages to JSON to feed to main
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_pages_data, f, indent=2, ensure_ascii=False)

print(f"[done] wrote {len(all_pages_data)} pages to {OUTPUT_FILE}")
