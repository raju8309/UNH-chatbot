# academic_standards_scrape.py
# Scrapes the UNH "Academic Standards" page into ordered JSON.

from bs4 import BeautifulSoup
import requests
import json
from urllib.parse import urljoin

URL = (
    "https://catalog.unh.edu/graduate/academic-regulations-degree-"
    "requirements/academic-standards/"
)
OUTPUT_FILE = "academic_standards.json"


def clean_text(tag):
    return tag.get_text(" ", strip=True)
  

def heading_level(name):
    if not name:
        return 0
    name = name.lower()
    if name in ("h2", "h3", "h4"):
        return int(name[1])
    return 0
  

def paragraph_starts_with_strong_heading(tag):
    if not tag.contents:
        return False
    first = tag.contents[0]
    return getattr(first, "name", None) == "strong"

  
print("[info] fetching page...")
try:
    resp = requests.get(URL, timeout=30)
    html = resp.text
except Exception as e:
    print("[error] could not download:", e)
    raise SystemExit(1) from e

  
soup = BeautifulSoup(html, "lxml")
main = (
    soup.find("main") or soup.find("article") or
    soup.find("div", id="content") or soup
)
flow = main.select("h2, h3, h4, p, ul, ol")

root = {"type": "root", "content": []}
stack = [(1, root)]
toc = []
  

def push_section(level, section_title):
    while stack and stack[-1][0] >= level:
        stack.pop()
    parent = stack[-1][1]
    new_node = {"type": "section", "title": section_title, "content": []}
    new_node["_current_inline_sub"] = None
    parent["content"].append(new_node)
    stack.append((level, new_node))
  

def add_paragraph(tag):
    _, current = stack[-1]

    # if paragraph begins with <strong>, start new subsection
    if paragraph_starts_with_strong_heading(tag):
        strong = tag.contents[0]
        sub_title = clean_text(strong).rstrip(":").strip()
        strong.extract()
        same_line_text = clean_text(tag)

        sub_node = {"type": "section", "title": sub_title, "content": []}
        if same_line_text:
            text_item = {"type": "text", "text": same_line_text}
            links = [
                {
                    "label": clean_text(a),
                    "url": urljoin(URL, a["href"].strip())
                }
                for a in tag.find_all("a", href=True)
            ]
            if links:
                text_item["links"] = links
            sub_node["content"].append(text_item)

        current["content"].append(sub_node)
        current["_current_inline_sub"] = sub_node
        return

    # normal paragraph
    target = (
        current["_current_inline_sub"]["content"]
        if current.get("_current_inline_sub")
        else current["content"]
    )
    item = {"type": "text", "text": clean_text(tag)}
    links = [
        {"label": clean_text(a), "url": urljoin(URL, a["href"].strip())}
        for a in tag.find_all("a", href=True)
    ]
    if links:
        item["links"] = links
    target.append(item)
  

def add_list(tag):
    # flatten list items into one paragraph string
    _, current = stack[-1]
    items = [clean_text(li) for li in tag.find_all("li", recursive=False)]
    flat_text = " • ".join(items)
    target = (
        current["_current_inline_sub"]["content"]
        if current.get("_current_inline_sub")
        else current["content"]
    )
    target.append({"type": "text", "text": flat_text})

  
print("[info] walking page...")
for el in flow:
    lvl = heading_level(el.name)
    if lvl:
        heading_title = clean_text(el)
        if lvl == 2:
            toc.append(heading_title)
        push_section(lvl, heading_title)
    else:
        if el.name == "p":
            add_paragraph(el)
        elif el.name in ("ul", "ol"):
            add_list(el)
  

def normalize(section_node):
    section_node.pop("_current_inline_sub", None)
    title = section_node["title"]
    text_blocks, link_bucket, children = [], [], []

    for item in section_node["content"]:
        if item["type"] == "text":
            text_blocks.append(item["text"])
            if "links" in item:
                link_bucket.extend(item["links"])
        elif item["type"] == "section":
            children.append(normalize(item))

    out = {"title": title}
    if text_blocks:
        out["text"] = text_blocks
    if link_bucket:
        out["links"] = link_bucket
    if children:
        out["subsections"] = children
    return out

  
print("[info] assembling JSON...")
page_title = (
    clean_text(soup.find("h1")) if soup.find("h1") else "Academic Standards"
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
