#!/usr/bin/env python3

import json
import os
import time
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from pdf_outline_extractor import make_outline
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_contextual_terms(persona: str, job: str) -> List[str]:
    """Extract meaningful terms from persona and job context"""
    combined_text = f"{persona} {job}".lower()
    
    # Enhanced stop words (minimal but effective)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
    }
    
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    meaningful_words = [w for w in words if w not in stop_words]
    
    # Use frequency to identify important terms
    word_freq = Counter(meaningful_words)
    important_terms = [word for word, count in word_freq.most_common(12)]
    
    return important_terms

def create_multi_perspective_query(persona: str, job: str) -> Tuple[List[str], List[str]]:
    """Create multiple query perspectives for richer semantic matching"""
    
    contextual_terms = extract_contextual_terms(persona, job)
    
    # Generate multiple query formulations (pure dynamic approach)
    queries = [
        f"{persona}: {job}",  # Direct query
        f"As a {persona}, I need to {job}",  # First person perspective
        f"Information relevant to {persona} who wants to {job}",  # Third person
        f"Specific details about {' '.join(contextual_terms[:5])}",  # Context-focused
        f"Practical information for {job}",  # Action-focused
        f"How to {job} as {persona}",  # Process-oriented
        f"{persona} looking for {' '.join(contextual_terms[:3])} information"  # Need-focused
    ]
    
    return queries, contextual_terms

def calculate_multi_query_similarity(section_text: str, query_embeddings: List[np.ndarray]) -> float:
    """Calculate similarity using multiple query perspectives"""
    
    section_embed = MODEL.encode([section_text])[0]
    
    # Calculate similarity with each query perspective
    similarities = [cosine_similarity(section_embed, query_embed) for query_embed in query_embeddings]
    
    # Use weighted combination of similarities
    max_similarity = max(similarities)  # Best match
    avg_similarity = sum(similarities) / len(similarities)  # Overall relevance
    top_3_avg = sum(sorted(similarities, reverse=True)[:3]) / 3  # Top matches
    
    # Weighted combination emphasizing best matches
    final_similarity = 0.5 * max_similarity + 0.3 * top_3_avg + 0.2 * avg_similarity
    
    return final_similarity

def assess_section_information_density(section: Dict) -> float:
    """Assess information density of section content"""
    content = section.get("content", "")
    
    if not content or len(content) < 100:
        return 0.0
    
    # Count information-rich elements
    density_score = 0.0
    
    # Structured content (lists, bullets)
    list_markers = len(re.findall(r'[•\-\*]\s+|\n\s*\d+\.\s+', content))
    density_score += min(list_markers * 0.05, 0.3)
    
    # Specific information (names, places, numbers)
    proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content))
    density_score += min(proper_nouns * 0.02, 0.2)
    
    # Numbers and measurements
    numbers = len(re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|€|$|km|hours?|days?|minutes?)\b', content))
    density_score += min(numbers * 0.03, 0.15)
    
    # Actionable language
    action_words = ['visit', 'try', 'explore', 'go', 'see', 'book', 'call', 'check', 'find']
    action_count = sum(content.lower().count(word) for word in action_words)
    density_score += min(action_count * 0.02, 0.2)
    
    # Normalize by content length
    length_factor = min(len(content) / 1000, 1.0)
    
    return density_score * length_factor

def create_rich_section_context(section: Dict) -> str:
    """Create rich contextual representation of section"""
    
    title = section["title"]
    content = section["content"]
    document = section["document"]
    
    # Use substantial content for better context (increased from previous versions)
    content_preview = content[:2500]  # More context for better matching
    
    # Create structured representation that helps with semantic matching
    section_context = f"""
    Document: {document}
    Section Title: {title}
    Content Summary: {content_preview}
    Information Type: {assess_section_information_density(section):.2f} density
    """.strip()
    
    return section_context

def process_document_with_content(pdf_path: str) -> List[Dict]:
    """Process a single PDF and extract sections with content"""
    result = make_outline(pdf_path)
    return result["sections"]

def rank_sections_pure_dynamic(persona: str, job: str, sections: List[Dict]) -> List[Dict]:
    """Pure dynamic ranking based on semantic similarity and content quality"""
    
    print(f"Pure dynamic ranking for: {persona} | {job}", flush=True)
    
    # Create multiple query perspectives
    queries, contextual_terms = create_multi_perspective_query(persona, job)
    print(f"Generated {len(queries)} query perspectives", flush=True)
    print(f"Key contextual terms: {contextual_terms[:8]}", flush=True)
    
    # Embed all queries
    query_embeddings = MODEL.encode(queries)
    
    # Process each section
    valid_sections = []
    for section in sections:
        # Basic quality filter
        content = section.get("content", "")
        if len(content) < 100:  # Skip very short sections
            continue
        
        # Create rich section context
        section_context = create_rich_section_context(section)
        
        # Calculate multi-perspective similarity
        similarity_score = calculate_multi_query_similarity(section_context, query_embeddings)
        
        # Add information density bonus
        density_bonus = assess_section_information_density(section) * 0.1
        
        # Final dynamic score
        section["dynamic_score"] = similarity_score + density_bonus
        valid_sections.append(section)
    
    # Sort purely by dynamic score
    ranked_sections = sorted(valid_sections, key=lambda x: x["dynamic_score"], reverse=True)
    
    print(f"Ranked {len(ranked_sections)} sections", flush=True)
    return ranked_sections

def extract_dynamic_subsections(sections: List[Dict], persona: str, job: str) -> List[Dict]:
    """Extract subsections using pure dynamic similarity"""
    
    all_subsections = []
    
    # Create specific query for subsection matching
    subsection_queries = [
        f"Specific actionable information for {persona} to {job}",
        f"Detailed recommendations relevant to {persona}",
        f"Practical steps for {job}",
    ]
    
    query_embeddings = MODEL.encode(subsection_queries)
    
    for section in sections:
        content = section["content"]
        
        if len(content) < 200:
            continue
        
        # Try multiple splitting strategies dynamically
        splitting_methods = [
            # Method 1: Structured content (bullets, numbers)
            lambda txt: [s.strip() for s in re.split(r'\n[•\-\*]\s+', txt) if len(s.strip()) > 80],
            # Method 2: Numbered items
            lambda txt: [s.strip() for s in re.split(r'\n\s*\d+\.\s+', txt) if len(s.strip()) > 80],
            # Method 3: Paragraph breaks
            lambda txt: [s.strip() for s in txt.split('\n\n') if len(s.strip()) > 100],
            # Method 4: Natural breaks (double newlines + capital letters)
            lambda txt: [s.strip() for s in re.split(r'\n\n(?=[A-Z])', txt) if len(s.strip()) > 80],
            # Method 5: Sentence grouping (fallback)
            lambda txt: [' '.join(txt.split()[i:i+50]) for i in range(0, len(txt.split()), 40) if len(txt.split()[i:i+50]) > 20]
        ]
        
        best_splits = []
        for method in splitting_methods:
            try:
                splits = method(content)
                if len(splits) >= 2 and len(splits) <= 10:  # Good split found
                    best_splits = splits
                    break
            except:
                continue
        
        # Score each subsection candidate
        if best_splits:
            # Limit for performance
            candidates = best_splits[:8]
            
            # Batch embedding for efficiency
            candidate_embeddings = MODEL.encode(candidates)
            
            for i, candidate_text in enumerate(candidates):
                # Calculate similarity to subsection queries
                similarities = [
                    cosine_similarity(candidate_embeddings[i], query_embed) 
                    for query_embed in query_embeddings
                ]
                
                # Use best match across query types
                best_similarity = max(similarities)
                
                # Weight by parent section relevance
                weighted_score = best_similarity * section["dynamic_score"]
                
                # Add information density bonus
                density_bonus = assess_section_information_density({"content": candidate_text}) * 0.1
                
                final_score = weighted_score + density_bonus
                
                all_subsections.append({
                    "document": section["document"],
                    "page": section["page"],
                    "parent_title": section["title"],
                    "text": candidate_text[:500],  # Limit text length
                    "dynamic_score": final_score
                })
    
    return sorted(all_subsections, key=lambda x: x["dynamic_score"], reverse=True)

def process_collection_pure_dynamic(input_dir: str, persona: str, job: str) -> Dict:
    """Pure dynamic processing without any hardcoded assumptions"""
    
    start_time = time.time()
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    all_sections = []
    
    print(f"Pure dynamic processing of {len(pdf_files)} documents", flush=True)
    print(f"Persona: '{persona}'", flush=True)
    print(f"Job: '{job}'", flush=True)
    
    # Process each PDF
    for pdf in pdf_files:
        if time.time() - start_time > 55:  # Time limit with buffer
            print("Time limit approaching, stopping processing", flush=True)
            break
        try:
            sections = process_document_with_content(str(pdf))
            all_sections.extend(sections)
            print(f"Processed {pdf.name}: {len(sections)} sections", flush=True)
        except Exception as e:
            print(f"Error processing {pdf.name}: {str(e)}", flush=True)
    
    print(f"Total sections extracted: {len(all_sections)}", flush=True)
    
    # Pure dynamic ranking
    ranked_sections = rank_sections_pure_dynamic(persona, job, all_sections)
    top_sections = ranked_sections[:10]
    
    print("Top sections (pure dynamic ranking):", flush=True)
    for i, section in enumerate(top_sections[:5]):
        print(f"  {i+1}. {section['title']} (score: {section['dynamic_score']:.4f})", flush=True)
    
    # Dynamic subsection extraction
    top_subsections = extract_dynamic_subsections(top_sections[:5], persona, job)[:15]
    
    print(f"Extracted {len(top_subsections)} top subsections", flush=True)
    
    # Generate timestamp
    timestamp = datetime.now().isoformat()
    
    return {
        "metadata": {
            "input_documents": [p.name for p in pdf_files],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp
        },
        "extracted_sections": [
            {
                "document": s["document"],
                "section_title": s["title"],
                "importance_rank": i + 1,
                "page_number": s["page"]
            } for i, s in enumerate(top_sections)
        ],
        "subsection_analysis": [
            {
                "document": sub["document"],
                "refined_text": sub["text"],
                "page_number": sub["page"]
            } for sub in top_subsections
        ]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pure Dynamic Persona-Driven Document Intelligence - Final Version')
    parser.add_argument('--input_dir', required=True, help='Directory with PDF documents')
    parser.add_argument('--persona', required=True, help='Persona description')
    parser.add_argument('--job', required=True, help='Job-to-be-done')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    result = process_collection_pure_dynamic(args.input_dir, args.persona, args.job)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Pure dynamic processing complete. Results saved to {args.output}")
    print(f"Final results: {len(result['extracted_sections'])} sections, {len(result['subsection_analysis'])} subsections")
