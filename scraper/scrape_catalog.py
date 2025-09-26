import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import openpyxl

# config
BASE_URL = "https://catalog.unh.edu"
SITEMAP_FILE = "graduate_catalog.xlsx"
OUTPUT_FILE = "unh_catalog.json"
ALLOWED_DOMAIN = "catalog.unh.edu"
TAB_IDS = ["#overviewtext", "#requirementstext", "#coursetext"]

# helpers
def clean_text(tag):
    return tag.get_text(" ", strip=True)

def heading_level(tag_name):
    if tag_name and tag_name.lower() in ("h2", "h3", "h4"):
        return int(tag_name[1])
    return 0

def push_section(level, section_title, stack, page_url):
    while stack and stack[-1][0] >= level:
        stack.pop()
    parent = stack[-1][1]
    new_node = {"type": "section", "title": section_title, "content": [], "page_url": page_url}
    parent["content"].append(new_node)
    stack.append((level, new_node))

def add_paragraph(tag, stack, page_url):
    _, current = stack[-1]
    text = clean_text(tag)
    if not text:  # skip empty
        return
    item = {"type": "text", "text": text}
    # Only include links within allowed domain
    links = [
        {"label": clean_text(a), "url": urljoin(page_url, a["href"].strip())}
        for a in tag.find_all("a", href=True)
        if ALLOWED_DOMAIN in urljoin(page_url, a["href"].strip())
    ]
    if links:
        item["links"] = links
    current["content"].append(item)

def add_list(tag, stack, page_url):
    _, current = stack[-1]
    items = [clean_text(li) for li in tag.find_all("li", recursive=False) if clean_text(li)]
    if not items:
        return
    item = {"type": "list", "items": items}
    links = [
        {"label": clean_text(a), "url": urljoin(page_url, a["href"].strip())}
        for a in tag.find_all("a", href=True)
        if ALLOWED_DOMAIN in urljoin(page_url, a["href"].strip())
    ]
    if links:
        item["links"] = links
    current["content"].append(item)

def normalize(section_node):
    title = section_node["title"]
    page_url = section_node.get("page_url", "")
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
    out = {"title": title, "page_url": page_url}
    if text_blocks:
        out["text"] = text_blocks
    if list_blocks:
        out["lists"] = list_blocks
    if link_bucket:
        out["links"] = link_bucket
    if children:
        out["subsections"] = children
    return out

def fetch_and_parse(url):
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"[error] could not download {url}: {e}")
        return None

# load sitemap
wb = openpyxl.load_workbook(SITEMAP_FILE)
sheet = wb.active
rows = list(sheet.iter_rows(min_row=2, values_only=True))  # skip header

merged = {"type": "root", "sections": []}

for row in rows:
    titles = [str(c).strip() for c in row[:-1] if c]
    relative_url = str(row[-1]).strip()
    if not relative_url:
        continue

    full_title = " > ".join(titles)
    URL = urljoin(BASE_URL, relative_url)
    print(f"[info] fetching page: {URL} ({full_title})")

    soup = fetch_and_parse(URL)
    if not soup:
        continue

    # determine main content container
    main = soup.find("main") or soup.find("article") or soup.find("div", id="content") or soup
    flow = main.select("h2, h3, h4, p, ul, ol")

    root = {"type": "root", "content": []}
    stack = [(1, root)]

    # parse main page content
    for el in flow:
        lvl = heading_level(el.name)
        if lvl:
            heading_title = clean_text(el)
            push_section(lvl, heading_title, stack, URL)
        else:
            if el.name == "p":
                add_paragraph(el, stack, URL)
            elif el.name in ("ul", "ol"):
                add_list(el, stack, URL)

    # fetch program/course tabs
    if "programs-study" in URL:
        for tab_id in TAB_IDS:
            tab_url = URL + tab_id
            tab_soup = fetch_and_parse(tab_url)
            if not tab_soup:
                continue
            tab_main = tab_soup.find("main") or tab_soup.find("article") or tab_soup.find("div", id="content") or tab_soup
            tab_flow = tab_main.select("h2, h3, h4, p, ul, ol")
            for el in tab_flow:
                lvl = heading_level(el.name)
                if lvl:
                    heading_title = clean_text(el)
                    push_section(lvl, heading_title, stack, tab_url)
                else:
                    if el.name == "p":
                        add_paragraph(el, stack, tab_url)
                    elif el.name in ("ul", "ol"):
                        add_list(el, stack, tab_url)

    sections = [normalize(n) for n in root["content"] if n.get("type") == "section"]
    merged["sections"].extend(sections)

# merge seperate files into one for simplicity and formatting in testing and main
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"[done] wrote merged catalog to {OUTPUT_FILE}")
