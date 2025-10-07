import requests
from bs4 import BeautifulSoup, NavigableString, Comment
import json
from urllib.parse import urljoin, urlparse
import openpyxl
import re

# config
BASE_URL = "https://catalog.unh.edu"
SITEMAP_FILE = "graduate_catalog.xlsx"   # run from project root; or change to "scraper/graduate_catalog.xlsx"
OUTPUT_FILE = "unh_catalog.json"         # output will be created alongside the script's CWD
ALLOWED_DOMAIN = "catalog.unh.edu"
# all program + course tabs
TAB_IDS = [
    "text", "overviewtext", "programstext", "coursetext",
    "coursestext", "facultytext", "requirementstext",
    "descriptiontext", "requirementstext", "acceleratedmasterstext",
    "studentlearningoutcomestext"]

# ---------- helpers ----------
def clean_text(tag_or_str):
    """Normalize spaces (e.g., \xa0) and collapse runs of whitespace."""
    if hasattr(tag_or_str, "get_text"):
        txt = tag_or_str.get_text(" ", strip=True)
    else:
        txt = str(tag_or_str or "")
    return re.sub(r"\s+", " ", txt).replace("\u00a0", " ").strip()

def heading_level(tag_name):
    if tag_name and tag_name.lower() in ("h2", "h3", "h4"):
        return int(tag_name[1])
    return 0

def dedupe_links(links):
    """De-duplicate link dicts by (label,url)."""
    seen = set()
    out = []
    for lk in links or []:
        label = lk.get("label", "").strip()
        url = lk.get("url", "").strip()
        key = (label, url)
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": label, "url": url})
    return out

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
    if not text:
        return
    item = {"type": "text", "text": text}
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
    """Flatten a parsed section node to the JSON structure you were using."""
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
        out["links"] = dedupe_links(link_bucket)
    if children:
        out["subsections"] = children

    # optional: attach course fields if this looks like a course section
    course_fields = extract_course_fields(out)
    if course_fields:
        out["course"] = course_fields

    return out

def fetch_and_parse(url):
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # preprocess divs to wrap floating text in <p> tags
        # specifically happens in course pages, but could be elsewhere too
        for element in soup.find_all("div"):
            for child in list(element.children):
                # NavigableString can be a comment, which we don't want... course pages have comments
                if isinstance(child, NavigableString) and not isinstance(child, Comment) and child.strip():
                    new_p = soup.new_tag("p")
                    new_p.string = child.strip()
                    child.replace_with(new_p)
        return soup
    except Exception as e:
        print(f"[error] could not download {url}: {e}")
        return None

def guess_page_title(soup, fallback):
    # Prefer on-page H1 if present; otherwise use fallback (from Excel titles chain)
    h1 = soup.find("h1")
    if h1:
        t = clean_text(h1)
        if t:
            return t
    return fallback

def infer_tier(url: str) -> int:
    """
    Decide tier by URL path:
      1 = Academic regulations & degree requirements
      2 = General info (and search)
      3 = Course descriptions
      4 = Programs of study
    NOTE: Order matters. We check for the most specific/unique paths first.
    """
    path = urlparse(url).path.lower()
    full = url.lower()

    # Tier 3: Course Descriptions (check first so they never fall through)
    if "/graduate/course-descriptions/" in path:
        return 3

    # Tier 4: Programs of Study
    if "/graduate/programs-study/" in path:
        return 4

    # Tier 1: Academic Regulations & Degree Requirements
    if "/graduate/academic-regulations-degree-requirements/" in path:
        return 1

    # Tier 2: General Information or search pages
    if "/graduate/general-information/" in path or "/search/?" in full or "/search/" in path:
        return 2

    # Default → treat as general info
    return 2

def infer_program_level(page_title: str, page_url: str, sections) -> str:
    """
    Infer program level from title; if unknown, backfill from URL tokens;
    if still unknown, scan early section text for degree abbreviations.
    """
    t = (page_title or "").lower()
    # 1) Title-based
    if re.search(r"\b(ph\.?d\.?|phd|doctor(al|ate)|dnp)\b", t):
        return "phd"
    if re.search(r"\b(ms|m\.?s\.?|ma|m\.?a\.?|m\.?sc|mba|mph|mfa|meng|m\.?eng)\b", t):
        return "grad"
    if re.search(r"\b(bs|b\.?s\.?|ba|b\.?a\.?)\b", t):
        return "undergrad"
    if "certificate" in t:
        return "certificate"

    # 2) URL-based backfill
    p = page_url.lower()
    # look for /phd/ -phd- etc.
    if re.search(r"(\b|/|-)(phd|ph\.?d\.?)(\b|/|-)", p):
        return "phd"
    if re.search(r"(\b|/|-)(dnp)(\b|/|-)", p):
        return "phd"
    if re.search(r"(\b|/|-)(ms|m-?s|ma|m-?a|mba|mph|mfa|meng|m-?eng)(\b|/|-)", p):
        return "grad"
    if re.search(r"(\b|/|-)(bs|b-?s|ba|b-?a)(\b|/|-)", p):
        return "undergrad"
    if "certificate" in p:
        return "certificate"

    # 3) First section text scan
    for sec in sections or []:
        for line in sec.get("text", [])[:3]:
            low = line.lower()
            if re.search(r"\bph\.?d\.?\b", low) or "doctoral" in low or "doctor of" in low:
                return "phd"
            if re.search(r"\b(m\.?s\.?|ms|mba|mph|mfa|m\.?eng|meng)\b", low):
                return "grad"
            if re.search(r"\b(b\.?s\.?|bs|b\.?a\.?|ba)\b", low):
                return "undergrad"
            if "certificate" in low:
                return "certificate"

    return "unknown"

def extract_program_aliases(page_title: str, page_url: str):
    """
    Generate robust aliases for program pages — no hand-coded list needed.
    E.g., "Information Technology (MS)" -> ["information technology", "ms", "ms it", "ms in information technology", "it"]
    """
    aliases = set()
    title = page_title or ""
    t = title

    # Basic normalized forms
    base = re.sub(r"\s+", " ", t).strip()
    base = re.sub(r"[–—\-]+", "-", base)  # normalize dashes
    base_no_paren = re.sub(r"\([^)]*\)", "", base).strip()

    def norm(s):
        return re.sub(r"\s+", " ", s).strip().lower()

    if base:
        aliases.add(norm(base))
    if base_no_paren and base_no_paren != base:
        aliases.add(norm(base_no_paren))

    # Pull acronym from parentheses e.g., "(MS)", "(MPH)", "(Ph.D.)"
    paren = re.findall(r"\(([^)]{1,30})\)", t)
    for p in paren:
        aliases.add(norm(p))

    # if title matches "<Name>, MS" style
    comma_deg = re.findall(r",\s*([A-Za-z\.\s]+)$", t)
    for p in comma_deg:
        aliases.add(norm(p))

    # tokenization and simple degree/program combos
    deg_keywords = {"ms", "m.s.", "ma", "m.a.", "mba", "mph", "phd", "ph.d.", "mfa", "meng", "m.eng", "bs", "b.s.", "ba", "b.a."}
    words = re.split(r"[\s,()/\-]+", t.lower())
    words = [w for w in words if w]
    # remove punctuation
    words = [re.sub(r"[^\w\.]", "", w) for w in words if w]
    # base phrase without known degree tokens
    core = [w for w in words if w not in deg_keywords]
    if core:
        core_phrase = " ".join(core).strip()
        if core_phrase:
            aliases.add(core_phrase)
            # short IT-ish aliases
            if "information" in core or "technology" in core:
                aliases.add("it")
                aliases.add("information technology")

    # build degree + core combos
    degs = [w for w in words if w in {"ms", "m.s.", "ma", "m.a.", "mba", "mph", "phd", "ph.d.", "mfa", "meng", "m.eng"}]
    if core:
        for d in degs:
            d_norm = d.replace(".", "")
            aliases.add(f"{d_norm} {' '.join(core)}".strip())
            if len(core) <= 3:
                # MS IT / MS CS style
                initialism = "".join([w[0] for w in core if w and w[0].isalpha()])
                if initialism:
                    aliases.add(f"{d_norm} {initialism}")
            aliases.add(f"{d_norm} in {' '.join(core)}")

    # URL tokens as extra fallbacks
    path = urlparse(page_url).path.lower()
    slug = path.strip("/").split("/")[-1]
    slug_tokens = [w for w in re.split(r"[\-_/]+", slug) if w]
    if slug_tokens:
        aliases.add(" ".join(slug_tokens))
        # pull trailing degree token like "...-ms"
        if slug_tokens[-1] in {"ms", "ma", "mba", "mph", "phd", "mfa"}:
            aliases.add(slug_tokens[-1])

    # collapse whitespace, lower
    aliases = {re.sub(r"\s+", " ", a).strip().lower() for a in aliases if a}
    # drop ultra-short single letters that are not helpful, except 'it'
    aliases = {a for a in aliases if len(a) > 1 or a == "it"}

    return sorted(aliases)

def extract_course_fields(section_dict):
    """
    If a section looks like a course (e.g., 'MATH 954 - Analysis II'), extract:
      - course_code: 'MATH 954'
      - credits: integer if we see 'Credits: X'
      - grade_mode: 'Letter Grading' etc.
    """
    title = section_dict.get("title", "")
    text_lines = section_dict.get("text", []) or []
    if not title:
        return None

    # Course code in title: e.g., "MATH 954 - Analysis II"
    m = re.match(r"^\s*([A-Z]{2,5}\s?\d{3,4}[A-Z]?)\b", title)
    course_code = m.group(1).replace("  ", " ") if m else None

    # Extract credits and grade mode from text lines
    credits = None
    grade_mode = None
    for line in text_lines:
        low = line.lower()
        mcr = re.search(r"\bcredits?\s*:\s*([0-9]+)", low)
        if mcr:
            try:
                credits = int(mcr.group(1))
            except Exception:
                pass
        mgm = re.search(r"\bgrade\s*mode\s*:\s*([A-Za-z ]+)", line)
        if mgm:
            grade_mode = clean_text(mgm.group(1))

    if course_code or credits is not None or grade_mode:
        return {"course_code": course_code, "credits": credits, "grade_mode": grade_mode}
    return None

# ---------- run ----------
# load sitemap
wb = openpyxl.load_workbook(SITEMAP_FILE)
sheet = wb.active
rows = list(sheet.iter_rows(min_row=2, values_only=True))  # skip header

merged = {"pages": []}

for row in rows:
    titles = [str(c).strip() for c in row[:-1] if c]
    relative_url = str(row[-1]).strip()
    if not relative_url:
        continue

    full_title = " > ".join(titles)  # breadcrumb-ish fallback
    URL = urljoin(BASE_URL, relative_url)
    print(f"[info] fetching page: {URL} ({full_title})")

    soup = fetch_and_parse(URL)
    if not soup:
        continue

    # figure out a good page title
    page_title = guess_page_title(soup, full_title)

    # determine main content container
    main = soup.find("main") or soup.find("article") or soup.find("div", id="content") or soup
    flow = main.select("h2, h3, h4, p, ul, ol")

    root = {"type": "root", "content": []}
    stack = [(1, root)]

    # create a page section
    push_section(2, full_title, stack, URL)

    # parse main page content
    for el in flow:
        lvl = heading_level(el.name)
        if lvl:
            heading_title = clean_text(el)
            push_section(lvl + 1, heading_title, stack, URL)
        else:
            if el.name == "p":
                add_paragraph(el, stack, URL)
            elif el.name in ("ul", "ol"):
                add_list(el, stack, URL)

    # fetch program/course tabs
    seen_panel_ids = set()
    for tab_id in TAB_IDS:
        if tab_id in seen_panel_ids:
            continue
        panel = soup.find(id=tab_id)
        if not panel:
            continue

        seen_panel_ids.add(tab_id)
        tab_flow = panel.select("h2, h3, h4, p, ul, ol")
        panel_url = f"{URL}#{tab_id}"

        # Push a new section so tab content is NOT made top-level; keeps
        # content contained, to help bot not mix up between programs/courses
        # Finds the current deepest level in the stack and adds tab at next level
        push_section(3, tab_id, stack, panel_url)

        for el in tab_flow:
            lvl = heading_level(el.name)
            if lvl:
                push_section(lvl + 2, clean_text(el), stack, panel_url)
            else:
                if el.name == "p":
                    add_paragraph(el, stack, panel_url)
                elif el.name in ("ul", "ol"):
                    add_list(el, stack, panel_url)

    sections = [normalize(n) for n in root["content"] if n.get("type") == "section"]

    # empty-page guard: if no sections parsed, create a basic section
    if not sections:
        page_body = clean_text(main)
        if page_body:
            sections = [{
                "title": "Page",
                "page_url": URL,
                "text": [page_body]
            }]

    # meta
    tier = infer_tier(URL)
    program_level = infer_program_level(page_title, URL, sections)

    page_record = {
        "page_title": page_title,
        "page_url": URL,
        "sections": sections,
        "meta": {
            "tier": tier,
            "program_level": program_level,
        },
    }

    # aliases:
    # - for programs (tier 4), generate program aliases
    # - for courses (tier 3), generate course aliases from section titles
    if tier == 4:
        page_record["aliases"] = extract_program_aliases(page_title, URL)
    elif tier == 3:
        # build aliases for each course-like section (SUBJ 123 / SUBJ123)
        course_aliases = set()
        for sec in sections:
            title = sec.get("title", "")
            m = re.match(r"^\s*([A-Z]{2,5})\s?(\d{3,4}[A-Z]?)\b", title)
            if m:
                subj, num = m.group(1), m.group(2)
                course_aliases.add(f"{subj}{num}".lower())
                course_aliases.add(f"{subj} {num}".lower())
        if course_aliases:
            page_record["aliases"] = sorted(course_aliases)

    merged["pages"].append(page_record)

# write output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"[done] wrote merged catalog to {OUTPUT_FILE} with {len(merged['pages'])} pages")
