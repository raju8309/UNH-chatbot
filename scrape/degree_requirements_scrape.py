from bs4 import BeautifulSoup
import requests
import json
from urllib.parse import urljoin

URL = (
    "https://catalog.unh.edu/graduate/academic-regulations-degree-"
    "requirements/degree-requirements/"
)
OUTPUT_FILE = "degree_requirements.json"

  
# clean up text inside tags
def clean_text(tag):
    return tag.get_text(" ", strip=True)

  
# check if tag is a heading (h2/h3/h4)
def heading_level(name):
    if not name:
        return 0
    name = name.lower()
    if name in ("h2", "h3", "h4"):
        return int(name[1])
    return 0

  
print("[info] fetching page...")
try:
    resp = requests.get(URL, timeout=30)
    html = resp.text
except Exception as e:
    print("[error] could not download:", e)
    raise SystemExit(1) from e

  
soup = BeautifulSoup(html, "lxml")
# try to grab main article content, fallback to whole page
main = (
    soup.find("main") or soup.find("article") or
    soup.find("div", id="content") or soup
)

  
# only keep headings + paragraphs + lists in order
flow = main.select("h2, h3, h4, p, ul, ol")

  
# build a tree using a stack (so subsections go under the right parent)
root = {"type": "root", "content": []}
stack = [(1, root)]
toc = []

  
def push_section(level, section_title):
    # pop until we find the parent level
    while stack and stack[-1][0] >= level:
        stack.pop()
    parent = stack[-1][1]
    new_node = {"type": "section", "title": section_title, "content": []}
    parent["content"].append(new_node)
    stack.append((level, new_node))

  
def add_paragraph(tag):
    _, current = stack[-1]
    item = {"type": "text", "text": clean_text(tag)}
    # collect any links inside the paragraph
    links = [
        {"label": clean_text(a), "url": urljoin(URL, a["href"].strip())}
        for a in tag.find_all("a", href=True)
    ]
    if links:
        item["links"] = links
    current["content"].append(item)

  
def add_list(tag):
    _, current = stack[-1]
    items = [clean_text(li) for li in tag.find_all("li", recursive=False)]
    item = {"type": "list", "items": items}
    # also collect links inside list items
    links = [
        {"label": clean_text(a), "url": urljoin(URL, a["href"].strip())}
        for a in tag.find_all("a", href=True)
    ]
    if links:
        item["links"] = links
    current["content"].append(item)

  
print("[info] walking page...")
for el in flow:
    lvl = heading_level(el.name)
    if lvl:  # if it's a heading
        heading_title = clean_text(el)
        if lvl == 2:
            toc.append(heading_title)  # top-level only
        push_section(lvl, heading_title)
    else:
        # normal content under last heading
        if el.name == "p":
            add_paragraph(el)
        elif el.name in ("ul", "ol"):
            add_list(el)

  
# turn tree into final JSON format (title, text, lists, links, subsections)
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
    out = {}
    out["title"] = title
    if text_blocks:
        out["text"] = text_blocks
    if list_blocks:
        out["lists"] = list_blocks
    if link_bucket:
        out["links"] = link_bucket
    if children:
        out["subsections"] = children
    return out

  
print("[info] assembling JSON...")
page_title = (
    clean_text(soup.find("h1")) if soup.find("h1") else "Degree Requirements"
)
sections = [
    normalize(n) for n in root["content"] if n.get("type") == "section"
]

  
data = {
    "page_title": page_title,
    "url": URL,
    "toc": toc,
    "sections": sections
}

  
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("[done] wrote:", OUTPUT_FILE)
