import json
import random
from collections import Counter

INPUT_FILE = "../scraper/unh_catalog.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_sections(sections):
    total = 0
    for s in sections:
        total += 1
        total += count_sections(s.get("subsections", []))
    return total

def collect_titles(sections):
    titles = []
    for s in sections:
        titles.append(s.get("title", ""))
        titles.extend(collect_titles(s.get("subsections", [])))
    return titles

def collect_links(sections):
    links = []
    for s in sections:
        links.extend(s.get("links", []))
        links.extend(collect_links(s.get("subsections", [])))
    return links

def main():
    data = load_json(INPUT_FILE)
    print(f"[info] Loaded {len(data)} pages from {INPUT_FILE}")

    # check for missing metadata
    missing_meta = [p for p in data if not p.get("page_title") or not p.get("url")]
    print(f"[check] Pages missing title/URL: {len(missing_meta)}")

    # count total sections and subsections
    total_sections = sum(count_sections(p["sections"]) for p in data)
    print(f"[check] Total sections (including subsections): {total_sections}")

    # count sections with text
    sections_with_text = sum(
        1 for p in data for s in p["sections"] if "text" in s
    )
    print(f"[check] Sections with text: {sections_with_text}")

    # collect all links
    all_links = []
    for p in data:
        all_links.extend(collect_links(p["sections"]))
    print(f"[check] Total links: {len(all_links)}")

    # most common link labels
    label_counts = Counter([link["label"] for link in all_links if "label" in link])
    print(f"[check] Top 5 most common link labels: {label_counts.most_common(5)}")

    # generate sample test questions
    questions = []
    for page in random.sample(data, min(20, len(data))):  # pick 20 random pages
        title = page["page_title"]
        questions.append(f"What are the requirements for {title}?")
        if page["sections"]:
            sec = random.choice(page["sections"])
            questions.append(f"What does the section '{sec['title']}' cover in {title}?")

    print("\n[tests] Sample chatbot questions:")
    for q in questions[:10]:
        print("-", q)

if __name__ == "__main__":
    main()
