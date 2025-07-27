#!/usr/bin/env python3
"""
PDF Outline Extractor - Enhanced Dynamic Version
Includes language-agnostic detection, content quality filtering, and performance optimizations
"""

import fitz  # PyMuPDF
import json
import sys
import time
import re
import unicodedata
import gc
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class TextBit:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_flags: int
    font_name: str

    @property
    def bold(self) -> bool:
        return bool(self.font_flags & 2**4)

    @property
    def italic(self) -> bool:
        return bool(self.font_flags & 2**1)

class PerfTracker:
    """Tracks performance and provides early stopping"""

    def __init__(self, max_time: float = 30.0):
        self.start = time.time()
        self.max_time = max_time
        self.stats = {
            "time_taken": 0,
            "pages_done": 0,
            "toc_heads": 0,
            "text_heads": 0,
            "final_heads": 0,
            "early_stop": False,
        }

    def log(self, msg: str) -> None:
        elapsed = time.time() - self.start
        print(f"[{elapsed:.2f}s] {msg}", file=sys.stderr)

    def time_left(self, buffer: float = 1.0) -> bool:
        elapsed = time.time() - self.start
        return elapsed < (self.max_time - buffer)

    def wrap_up(self) -> None:
        self.stats["time_taken"] = time.time() - self.start
        print(f"Stats: {self.stats}", file=sys.stderr)

def clean_text(text: str) -> str:
    """Enhanced text cleaning for multiple languages"""
    if not text:
        return ""

    # Normalize unicode (handles various language encodings)
    text = unicodedata.normalize("NFKC", text)
    
    # Remove directional marks and formatting characters
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    
    # Handle CJK spacing (Chinese, Japanese, Korean)
    text = re.sub(
        r"(?<=[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff])\s+(?=[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff])",
        "",
        text,
    )
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def assess_content_quality(text: str) -> float:
    """Dynamically assess content quality without language assumptions"""
    if not text or len(text) < 3:
        return 0.0
    
    # Universal quality metrics
    word_count = len(text.split())
    char_variety = len(set(text.lower().replace(' ', '')))
    
    # Quality indicators that work across languages
    quality_score = 0.0
    
    # Sufficient length and character variety
    if word_count >= 3 and char_variety >= 3:
        quality_score += 0.5
    
    # Not overly repetitive
    if char_variety / max(len(text.replace(' ', '')), 1) > 0.3:
        quality_score += 0.3
    
    # Reasonable word length distribution
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 2 <= avg_word_len <= 12:  # Universal reasonable range
            quality_score += 0.2
    
    return min(quality_score, 1.0)

def get_doc_title(doc: fitz.Document, pdf_path: str) -> str:
    """Enhanced title extraction with quality assessment"""
    # Try metadata first
    try:
        meta = doc.metadata
        if meta and meta.get("title", "").strip():
            title = clean_text(meta["title"])
            if 3 < len(title) < 200 and assess_content_quality(title) > 0.5:
                return title
    except:
        pass

    # Extract from first page with quality filtering
    try:
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]

            candidates = []
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_parts = []
                    line_sizes = []

                    for span in line["spans"]:
                        txt = clean_text(span.get("text", ""))
                        size = span.get("size", 0)

                        if txt and assess_content_quality(txt) > 0.3:
                            line_parts.append(txt)
                            line_sizes.append(size)

                    if line_parts and line_sizes:
                        combined = " ".join(line_parts)
                        avg_size = sum(line_sizes) / len(line_sizes)
                        
                        candidates.append({
                            'text': combined,
                            'size': avg_size,
                            'quality': assess_content_quality(combined)
                        })

            # Sort by size and quality
            if candidates:
                candidates.sort(key=lambda x: (x['size'], x['quality']), reverse=True)
                best_candidate = candidates[0]
                if 3 < len(best_candidate['text']) < 200:
                    return best_candidate['text']
    except:
        pass

    return Path(pdf_path).stem

def grab_toc_heads(doc: fitz.Document, tracker: PerfTracker) -> List[Dict]:
    """Extract TOC headings with quality filtering"""
    heads = []
    tracker.log("Grabbing TOC headings")

    try:
        toc = doc.get_toc()
        if not toc:
            return heads

        for lvl, title, pg in toc:
            if lvl > 3:  # Limit depth for performance
                continue

            txt = clean_text(title)
            if txt and len(txt) > 1 and assess_content_quality(txt) > 0.3:
                heads.append({
                    "level": f"H{lvl}",
                    "text": txt,
                    "page": max(1, pg),
                    "source": "toc",
                })

        tracker.stats["toc_heads"] = len(heads)

    except Exception as e:
        print(f"TOC grab failed: {e}", file=sys.stderr)

    return heads

def extract_texts_batch(doc: fitz.Document, start_page: int, end_page: int, tracker: PerfTracker) -> List[TextBit]:
    """Extract text elements from a page batch for memory efficiency"""
    elems = []
    
    for pg_num in range(start_page, end_page):
        if not tracker.time_left():
            tracker.stats["early_stop"] = True
            break
            
        try:
            page = doc[pg_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    spans = []
                    for span in line["spans"]:
                        txt = clean_text(span.get("text", ""))
                        if txt and assess_content_quality(txt) > 0.2:
                            spans.append({
                                'text': txt,
                                'x': span.get("bbox", [0, 0, 0, 0])[0],
                                'size': span.get("size", 12),
                                'flags': span.get("flags", 0),
                                'font': span.get("font", ""),
                                'bbox': span.get("bbox", (0, 0, 0, 0))
                            })
                    
                    if not spans:
                        continue
                    
                    spans.sort(key=lambda s: s['x'])
                    
                    parts = []
                    prev = None
                    
                    for span in spans:
                        if prev and span['x'] - prev['bbox'][2] > 5:
                            parts.append(" ")
                        parts.append(span['text'])
                        prev = span
                    
                    combined = "".join(parts).strip()
                    
                    if combined and len(combined) > 1:
                        avg_size = sum(s['size'] for s in spans) / len(spans)
                        common_flags = max(set(s['flags'] for s in spans), 
                                         key=lambda f: [s['flags'] for s in spans].count(f))
                        
                        elems.append(TextBit(
                            text=combined,
                            page=pg_num + 1,
                            bbox=spans[0]['bbox'],
                            font_size=avg_size,
                            font_flags=common_flags,
                            font_name=spans[0]['font']
                        ))
        
        except Exception as e:
            print(f"Page {pg_num + 1} error: {e}", file=sys.stderr)
            continue
    
    return elems

def extract_texts(doc: fitz.Document, tracker: PerfTracker, batch_size: int = 50) -> List[TextBit]:
    """Extract text elements with batching for large documents"""
    tracker.log("Pulling text elements with batching")
    
    total_pages = len(doc)
    all_elems = []
    
    # Process in batches for memory efficiency
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        
        batch_elems = extract_texts_batch(doc, batch_start, batch_end, tracker)
        all_elems.extend(batch_elems)
        
        # Clear batch memory
        del batch_elems
        gc.collect()
        
        if not tracker.time_left():
            break
    
    tracker.stats["pages_done"] = min(total_pages, len(all_elems))
    tracker.log(f"Got {len(all_elems)} text elements from {tracker.stats['pages_done']} pages")
    return all_elems

def extract_section_content_smart(doc: fitz.Document, start_page: int, end_page: int) -> str:
    """Extract section content with dynamic quality filtering"""
    content_chunks = []
    
    for page_num in range(start_page, end_page + 1):
        if page_num >= len(doc):
            break
            
        page_text = doc[page_num].get_text()
        
        # Dynamic quality assessment - filter out likely junk
        lines = page_text.split('\n')
        quality_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Universal quality metrics
            if assess_content_quality(line) > 0.3:
                quality_lines.append(line)
        
        if quality_lines:
            content_chunks.append('\n'.join(quality_lines))
    
    return '\n\n'.join(content_chunks)

def detect_heading_patterns_dynamic(txt: str) -> float:
    """Language-agnostic heading pattern detection"""
    if not txt:
        return 0.0
    
    score = 0.0
    
    # Universal patterns that work across languages
    
    # Pattern 1: Numbering systems (universal)
    if re.match(r'^\d+[\.\-\)]\s*', txt):
        score += 0.4
    elif re.match(r'^\d+\.\d+[\.\-\)]\s*', txt):
        score += 0.3
    
    # Pattern 2: Length-based (headings are typically short)
    word_count = len(txt.split())
    if 2 <= word_count <= 8:
        score += 0.3
    elif 1 <= word_count <= 12:
        score += 0.1
    
    # Pattern 3: Capitalization patterns (language-flexible)
    if txt[0].isupper() and not txt.endswith('.'):
        score += 0.2
    
    # Pattern 4: All caps (common for headings)
    if txt.isupper() and 3 <= len(txt) <= 50:
        score += 0.3
    
    # Pattern 5: Title case
    if txt.istitle() and len(txt.split()) <= 8:
        score += 0.2
    
    return min(score, 1.0)

def is_metadata_dynamic(txt: str, pg: int) -> bool:
    """Dynamic metadata detection without language assumptions"""
    
    # Universal metadata patterns
    universal_patterns = [
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$',  # Dates
        r'^https?://',  # URLs
        r'^www\.',  # Web addresses
        r'^\d+$',  # Page numbers
        r'^[A-Z]{2,6}\s*\d+$',  # Codes
    ]
    
    # More restrictive on first few pages
    if pg <= 3:
        return any(re.match(p, txt, re.IGNORECASE) for p in universal_patterns)
    
    # Less restrictive later in document
    return any(re.match(p, txt, re.IGNORECASE) for p in universal_patterns[:2])

def looks_like_heading_enhanced(txt: str, size: float, bold: bool, body_size: float, pg: int) -> bool:
    """Enhanced heading detection with dynamic patterns"""
    if not txt or len(txt.strip()) < 2:
        return False
    
    txt = txt.strip()
    words = len(txt.split())
    
    # Skip metadata
    if is_metadata_dynamic(txt, pg):
        return False
    
    # Quality assessment
    if assess_content_quality(txt) < 0.3:
        return False
    
    # Universal rejection criteria
    rejects = [
        len(txt) > 150 or words > 15,  # Too long
        re.match(r"^[.\-_=]{3,}$", txt),  # Just punctuation
        len(set(txt.replace(" ", ""))) < 3 and len(txt) > 8,  # Too repetitive
    ]
    
    if any(rejects):
        return False
    
    # Dynamic pattern detection
    pattern_score = detect_heading_patterns_dynamic(txt)
    
    # Font-based indicators
    ratio = size / body_size if body_size > 0 else 1.0
    font_score = 0.0
    
    if ratio > 1.15:
        font_score += 0.4
    if bold and ratio > 1.05:
        font_score += 0.3
    
    # Combined scoring
    total_score = pattern_score + font_score
    
    return total_score > 0.5

def get_level_dynamic(txt: str, size: float, bold: bool, body_size: float) -> str:
    """Dynamic level assignment based on content and formatting"""
    ratio = size / body_size if body_size > 0 else 1.0
    pattern_score = detect_heading_patterns_dynamic(txt)
    
    # Dynamic level assignment
    if ratio >= 1.4 or pattern_score >= 0.7:
        return "H1"
    elif ratio >= 1.2 or pattern_score >= 0.5:
        return "H2"
    else:
        return "H3"

def pull_headings(elems: List[TextBit], tracker: PerfTracker) -> List[Dict]:
    """Enhanced heading extraction with dynamic patterns"""
    tracker.log("Extracting headings with enhanced detection")
    
    if not elems:
        tracker.log("No elements to extract!")
        return []
    
    # Dynamic body size detection
    body_candidates = []
    for el in elems:
        if not el.bold and len(el.text.split()) > 5 and assess_content_quality(el.text) > 0.5:
            body_candidates.append(el.font_size)
    
    if body_candidates:
        # Use most common size as body size
        from collections import Counter
        size_counter = Counter(body_candidates)
        body_size = size_counter.most_common(1)[0][0]
    else:
        # Fallback
        all_sizes = [el.font_size for el in elems if not el.bold]
        body_size = max(set(all_sizes), key=all_sizes.count) if all_sizes else 12
    
    tracker.log(f"Dynamic body font size detected: {body_size:.1f}")
    
    heads = []
    seen = set()
    
    for el in elems:
        txt = el.text.strip()
        low_txt = txt.lower()
        
        if low_txt in seen:
            continue
        
        if looks_like_heading_enhanced(txt, el.font_size, el.bold, body_size, el.page):
            lvl = get_level_dynamic(txt, el.font_size, el.bold, body_size)
            
            heads.append({
                "level": lvl,
                "text": txt,
                "page": el.page,
                "source": "analysis",
            })
            
            seen.add(low_txt)
    
    heads.sort(key=lambda x: (x["page"], x["level"]))
    tracker.stats["text_heads"] = len(heads)
    tracker.log(f"Found {len(heads)} headings")
    return heads

def merge_heads(toc_heads: List[Dict], text_heads: List[Dict]) -> List[Dict]:
    """Merge headings with improved duplicate detection"""
    finals = []
    seen = set()

    # Process TOC headings first (usually more reliable)
    for head in toc_heads:
        norm = head["text"].lower().strip()
        if norm not in seen and len(head["text"]) > 2:
            finals.append({
                "level": head["level"],
                "text": head["text"],
                "page": head["page"],
            })
            seen.add(norm)

    # Process text headings with similarity checking
    for head in text_heads:
        norm = head["text"].lower().strip()

        if norm in seen:
            continue

        # Enhanced similarity detection
        is_similar = False
        for existing in seen:
            # Exact substring match
            if (norm in existing or existing in norm) and abs(len(norm) - len(existing)) < 10:
                is_similar = True
                break
            
            # Remove numbering and compare
            clean_norm = re.sub(r'^\d+\.?\s*', '', norm)
            clean_existing = re.sub(r'^\d+\.?\s*', '', existing)
            if clean_norm == clean_existing and len(clean_norm) > 5:
                is_similar = True
                break

        if not is_similar and len(head["text"]) > 2:
            finals.append({
                "level": head["level"],
                "text": head["text"],
                "page": head["page"],
            })
            seen.add(norm)

    finals.sort(key=lambda x: (x["page"], x["level"]))
    return finals

def make_outline(pdf_path: str) -> Dict:
    """Enhanced outline creation with all improvements"""
    tracker = PerfTracker(max_time=30.0)
    
    try:
        doc = fitz.open(pdf_path)
        tracker.log(f"PDF loaded, {len(doc)} pages")
        
        title = get_doc_title(doc, pdf_path)
        toc_heads = grab_toc_heads(doc, tracker)
        
        elems = extract_texts(doc, tracker)
        text_heads = pull_headings(elems, tracker)
        
        finals = merge_heads(toc_heads, text_heads)
        tracker.stats["final_heads"] = len(finals)
        
        sections = []
        for i, head in enumerate(finals):
            start_page = head["page"] - 1
            end_page = (
                finals[i+1]["page"] - 2
                if (i+1) < len(finals)
                else len(doc) - 1
            )
            
            # Use enhanced content extraction
            section_content = extract_section_content_smart(doc, start_page, end_page)
            sections.append({
                "document": Path(pdf_path).name,
                "page": head["page"],
                "title": head["text"],
                "content": section_content,
                "level": head["level"]
            })
        
        doc.close()
        del elems
        gc.collect()
        
        tracker.wrap_up()
        return {
            "title": title,
            "sections": sections
        }
        
    except Exception as e:
        print(f"PDF error: {e}", file=sys.stderr)
        raise

def handle_pdf(pdf_path: str, out_path: str):
    """Process one PDF and save to JSON"""
    try:
        res = make_outline(pdf_path)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        print(f"Done: {Path(pdf_path).name} -> {Path(out_path).name}")
        return True

    except Exception as e:
        print(f"Failed {Path(pdf_path).name}: {e}")
        return False

def main():
    """Batch process PDFs in input directory"""
    in_dir = Path("input")
    out_dir = Path("output")

    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(in_dir.glob("*.pdf"))

    if not pdfs:
        print("No PDFs in input dir")
        sys.exit(1)

    print(f"Processing {len(pdfs)} PDFs with enhanced extraction")

    successes = 0
    start = time.time()

    for pdf in pdfs:
        out_file = pdf.stem + ".json"
        out_path = out_dir / out_file

        if handle_pdf(str(pdf), str(out_path)):
            successes += 1

    total_t = time.time() - start

    print("\nDone:")
    print(f"  Success: {successes}/{len(pdfs)}")
    print(f"  Time: {total_t:.2f}s")
    print(f"  Output: {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and not os.path.exists("/app/input"):
        pdf_path = sys.argv[1]
        if not Path(pdf_path).is_file():
            print(f"PDF not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        
        try:
            res = make_outline(pdf_path)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        main()
