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

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_contextual_terms(persona: str, job: str) -> List[str]:
    """Extract meaningful terms from persona and job context"""
    combined_text = f"{persona} {job}".lower()

    # Enhanced stop words (minimal but effective)
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
    }

    # Extract meaningful words
    words = re.findall(r"\b[a-zA-Z]{3,}\b", combined_text)
    meaningful_words = [w for w in words if w not in stop_words]

    # Use frequency to identify important terms
    word_freq = Counter(meaningful_words)
    important_terms = [word for word, count in word_freq.most_common(12)]

    return important_terms


def create_multi_perspective_query(
    persona: str, job: str
) -> Tuple[List[str], List[str]]:
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
        f"{persona} looking for {' '.join(contextual_terms[:3])} information",  # Need-focused
    ]

    return queries, contextual_terms


def calculate_multi_query_similarity(
    section_text: str, query_embeddings: List[np.ndarray]
) -> float:
    """Calculate similarity using multiple query perspectives"""

    section_embed = MODEL.encode([section_text])[0]

    # Calculate similarity with each query perspective
    similarities = [
        cosine_similarity(section_embed, query_embed)
        for query_embed in query_embeddings
    ]

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
    list_markers = len(re.findall(r"[•\-\*]\s+|\n\s*\d+\.\s+", content))
    density_score += min(list_markers * 0.05, 0.3)

    # Specific information (names, places, numbers)
    proper_nouns = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content))
    density_score += min(proper_nouns * 0.02, 0.2)

    # Numbers and measurements
    numbers = len(
        re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|€|$|km|hours?|days?|minutes?)\b", content)
    )
    density_score += min(numbers * 0.03, 0.15)

    # Actionable language
    action_words = [
        "visit",
        "try",
        "explore",
        "go",
        "see",
        "book",
        "call",
        "check",
        "find",
    ]
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


def rank_sections_pure_dynamic(
    persona: str, job: str, sections: List[Dict]
) -> List[Dict]:
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
        similarity_score = calculate_multi_query_similarity(
            section_context, query_embeddings
        )

        # Add information density bonus
        density_bonus = assess_section_information_density(section) * 0.1

        # Final dynamic score
        section["dynamic_score"] = similarity_score + density_bonus
        valid_sections.append(section)

    # Sort purely by dynamic score
    ranked_sections = sorted(
        valid_sections, key=lambda x: x["dynamic_score"], reverse=True
    )

    print(f"Ranked {len(ranked_sections)} sections", flush=True)
    return ranked_sections


def extract_dynamic_subsections(
    sections: List[Dict], persona: str, job: str
) -> List[Dict]:
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
            lambda txt: [
                s.strip() for s in re.split(r"\n[•\-\*]\s+", txt) if len(s.strip()) > 80
            ],
            # Method 2: Numbered items
            lambda txt: [
                s.strip()
                for s in re.split(r"\n\s*\d+\.\s+", txt)
                if len(s.strip()) > 80
            ],
            # Method 3: Paragraph breaks
            lambda txt: [s.strip() for s in txt.split("\n\n") if len(s.strip()) > 100],
            # Method 4: Natural breaks (double newlines + capital letters)
            lambda txt: [
                s.strip()
                for s in re.split(r"\n\n(?=[A-Z])", txt)
                if len(s.strip()) > 80
            ],
            # Method 5: Sentence grouping (fallback)
            lambda txt: [
                " ".join(txt.split()[i : i + 50])
                for i in range(0, len(txt.split()), 40)
                if len(txt.split()[i : i + 50]) > 20
            ],
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
                density_bonus = (
                    assess_section_information_density({"content": candidate_text})
                    * 0.1
                )

                final_score = weighted_score + density_bonus

                all_subsections.append(
                    {
                        "document": section["document"],
                        "page": section["page"],
                        "parent_title": section["title"],
                        "text": candidate_text[:500],  # Limit text length
                        "dynamic_score": final_score,
                    }
                )

    return sorted(all_subsections, key=lambda x: x["dynamic_score"], reverse=True)


def analyze_pdf_collection(input_dir: str) -> Dict:
    """Analyze PDF collection to suggest title and context"""
    pdf_files = list(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        return {
            "suggested_title": "Document Collection",
            "pdf_count": 0,
            "pdf_names": [],
        }

    # Extract PDF names for analysis
    pdf_names = [pdf.stem for pdf in pdf_files]

    # Try to detect common themes from filenames
    all_words = []
    for name in pdf_names:
        # Clean filename and extract words
        words = re.findall(r"[A-Za-z]+", name)
        all_words.extend([w.lower() for w in words if len(w) > 2])

    # Find common themes
    word_freq = Counter(all_words)
    common_themes = [word for word, count in word_freq.most_common(5) if count > 1]

    # Generate suggested title
    if common_themes:
        suggested_title = " ".join(common_themes[:3]).title() + " Collection"
    elif len(pdf_names) == 1:
        suggested_title = pdf_names[0].replace("_", " ").replace("-", " ").title()
    else:
        suggested_title = "Document Collection"

    return {
        "suggested_title": suggested_title,
        "pdf_count": len(pdf_files),
        "pdf_names": [pdf.name for pdf in pdf_files],
        "common_themes": common_themes,
    }


def generate_challenge_info(themes: List[str], suggested_title: str) -> Dict:
    """Generate challenge info based on document themes"""

    # Generate challenge ID based on themes
    if any(
        theme in ["travel", "trip", "tourism", "france", "city", "south"]
        for theme in themes
    ):
        challenge_id = "round_1b_002"
        test_case_name = "travel_planner"
        description = extract_description_from_title(suggested_title) or "France Travel"
    elif any(
        theme in ["business", "finance", "investment", "market", "revenue"]
        for theme in themes
    ):
        challenge_id = "round_1b_003"
        test_case_name = "business_analyst"
        description = "Business Analysis"
    elif any(
        theme in ["research", "study", "academic", "science", "paper"]
        for theme in themes
    ):
        challenge_id = "round_1b_001"
        test_case_name = "academic_researcher"
        description = "Academic Research"
    elif any(theme in ["hr", "human", "employee", "onboarding"] for theme in themes):
        challenge_id = "round_1b_004"
        test_case_name = "hr_professional"
        description = "HR Management"
    else:
        challenge_id = "round_1b_999"
        test_case_name = "general_analysis"
        description = "Document Analysis"

    return {
        "challenge_id": challenge_id,
        "test_case_name": test_case_name,
        "description": description,
    }


def generate_documents_metadata(input_dir: str) -> List[Dict]:
    """Generate document metadata with filenames and titles"""
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    documents = []

    for pdf_file in sorted(pdf_files):
        filename = pdf_file.name

        # Extract clean title from filename
        title = extract_title_from_filename(filename)

        documents.append({"filename": filename, "title": title})

    return documents


def extract_title_from_filename(filename: str) -> str:
    """Extract clean title from PDF filename"""
    # Remove .pdf extension
    title = filename.replace(".pdf", "")

    # Replace underscores and hyphens with spaces
    title = title.replace("_", " ").replace("-", " ")

    # Clean up multiple spaces
    title = re.sub(r"\s+", " ", title).strip()

    # Capitalize appropriately
    title = title.title()

    return title


def extract_description_from_title(title: str) -> str:
    """Extract description from document collection title"""
    # Remove common collection words
    description = (
        title.replace(" Collection", "").replace(" Guide", "").replace(" Documents", "")
    )

    # Clean and format
    description = description.strip().title()

    return description


def suggest_persona_from_themes(themes: List[str]) -> str:
    """Suggest persona based on detected themes"""
    if any(
        theme in ["travel", "trip", "tourism", "city", "france", "south"]
        for theme in themes
    ):
        return "Travel Planner"
    elif any(theme in ["business", "finance", "investment"] for theme in themes):
        return "Investment Analyst"
    elif any(theme in ["research", "study", "academic"] for theme in themes):
        return "PhD Researcher"
    elif any(theme in ["hr", "human", "employee"] for theme in themes):
        return "HR Professional"
    elif any(theme in ["tech", "software", "development"] for theme in themes):
        return "Software Engineer"
    else:
        return "Document Analyst"


def suggest_task_from_themes_and_persona(themes: List[str], persona: str) -> str:
    """Suggest job-to-be-done based on themes and persona"""

    if "Travel Planner" in persona:
        if any(theme in ["france", "south", "guide"] for theme in themes):
            return "Plan a trip of 4 days for a group of 10 college friends."
        else:
            return "Plan a comprehensive travel itinerary based on the available information."

    elif "HR Professional" in persona:
        return (
            "Create and manage fillable forms for employee onboarding and compliance."
        )

    elif "Investment Analyst" in persona:
        return "Analyze revenue trends, R&D investments, and market positioning strategies."

    elif "PhD Researcher" in persona or "Researcher" in persona:
        return "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks."

    elif "Software Engineer" in persona:
        return (
            "Find implementation details and best practices for software development."
        )

    else:
        return "Analyze and extract key insights from the document collection for decision making."


def auto_create_input_json(input_dir: str, output_path: str = None) -> str:
    """Auto-create enhanced input JSON with challenge info and document metadata"""

    if output_path is None:
        output_path = Path(input_dir).parent / "challenge1b_input.json"

    print("🚀 Auto-generating Enhanced Input JSON")
    print("=" * 50)

    # Analyze PDF collection
    analysis = analyze_pdf_collection(input_dir)

    print(f"📁 Found {analysis['pdf_count']} PDF files:")
    for pdf_name in analysis["pdf_names"]:
        print(f"   • {pdf_name}")

    if analysis["common_themes"]:
        print(f"🔍 Detected themes: {', '.join(analysis['common_themes'])}")

    print(f"💡 Suggested title: {analysis['suggested_title']}")
    print()

    # Generate challenge info based on themes
    challenge_info = generate_challenge_info(
        analysis["common_themes"], analysis["suggested_title"]
    )

    print(f"🏆 Challenge Info:")
    print(f"   ID: {challenge_info['challenge_id']}")
    print(f"   Test Case: {challenge_info['test_case_name']}")
    print(f"   Description: {challenge_info['description']}")

    # Generate document metadata
    documents_metadata = generate_documents_metadata(input_dir)

    # Get persona
    print("\n👤 Step 1: Define your persona/role")
    persona_examples = [
        "Travel Planner",
        "HR Professional",
        "PhD Researcher",
        "Investment Analyst",
        "Software Engineer",
        "Marketing Manager",
        "Project Manager",
        "Student",
    ]

    print("   Examples:", ", ".join(persona_examples))

    # Suggest persona based on themes
    suggested_persona = suggest_persona_from_themes(analysis["common_themes"])
    if suggested_persona:
        print(f"   💡 Suggested persona: {suggested_persona}")

    persona = input("   Enter your persona: ").strip()

    if not persona:
        persona = suggested_persona or "Document Analyst"
        print(f"   Using suggested: {persona}")

    # Get job/task
    print("\n🎯 Step 2: Define your job-to-be-done")

    # Suggest tasks based on detected themes and persona
    task_suggestion = suggest_task_from_themes_and_persona(
        analysis["common_themes"], persona
    )
    print(f"   💡 Suggested task: {task_suggestion}")

    task = input(
        "   Enter your job-to-be-done (or press Enter to use suggestion): "
    ).strip()

    if not task:
        task = task_suggestion
        print(f"   Using suggested task: {task}")

    # Create enhanced JSON structure
    input_data = {
        "challenge_info": challenge_info,
        "documents": documents_metadata,
        "persona": {"role": persona},
        "job_to_be_done": {"task": task},
        "auto_generated": True,
        "generation_timestamp": datetime.now().isoformat(),
        "source_analysis": {
            "pdf_count": analysis["pdf_count"],
            "detected_themes": analysis["common_themes"],
        },
    }

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Enhanced Input JSON created successfully!")
    print(f"📁 Saved to: {output_path}")
    print("\n📄 Generated structure preview:")
    preview_data = {
        "challenge_info": input_data["challenge_info"],
        "documents": f"[{len(input_data['documents'])} documents]",
        "persona": input_data["persona"],
        "job_to_be_done": input_data["job_to_be_done"],
    }
    print(json.dumps(preview_data, indent=2))

    return str(output_path)


def process_collection_from_json(input_json: Dict, input_dir: str) -> Dict:
    """Process collection using JSON input format"""

    # Extract persona and job from JSON structure
    persona = input_json["persona"]["role"]
    job = input_json["job_to_be_done"]["task"]

    # Optional: Extract title for context boosting if available
    title_context = input_json.get("title", "")
    if "challenge_info" in input_json:
        title_context = input_json["challenge_info"].get("description", title_context)

    start_time = time.time()
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    all_sections = []

    print(f"Processing {len(pdf_files)} documents from JSON input", flush=True)
    print(f"Persona: '{persona}'", flush=True)
    print(f"Job: '{job}'", flush=True)
    if title_context:
        print(f"Title context: '{title_context}'", flush=True)

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

    # Create enhanced job description using title context if available
    enhanced_job = job
    if title_context:
        enhanced_job = f"{job}. Context: {title_context}"

    # Pure dynamic ranking
    ranked_sections = rank_sections_pure_dynamic(persona, enhanced_job, all_sections)
    top_sections = ranked_sections[:10]

    print("Top sections (pure dynamic ranking):", flush=True)
    for i, section in enumerate(top_sections[:5]):
        print(
            f"  {i+1}. {section['title']} (score: {section['dynamic_score']:.4f})",
            flush=True,
        )

    # Dynamic subsection extraction
    top_subsections = extract_dynamic_subsections(
        top_sections[:5], persona, enhanced_job
    )[:15]

    print(f"Extracted {len(top_subsections)} top subsections", flush=True)

    # Generate timestamp
    timestamp = datetime.now().isoformat()

    return {
        "metadata": {
            "input_documents": [p.name for p in pdf_files],
            "persona": persona,
            "job_to_be_done": job,
            "title_context": title_context,
            "processing_timestamp": timestamp,
            "auto_generated_input": input_json.get("auto_generated", False),
        },
        "extracted_sections": [
            {
                "document": s["document"],
                "section_title": s["title"],
                "importance_rank": i + 1,
                "page_number": s["page"],
            }
            for i, s in enumerate(top_sections)
        ],
        "subsection_analysis": [
            {
                "document": sub["document"],
                "refined_text": sub["text"],
                "page_number": sub["page"],
            }
            for sub in top_subsections
        ],
    }


def load_input_json(json_path: str) -> Dict:
    """Load and validate enhanced JSON input with challenge info and documents"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)

        # Validate required fields
        if "persona" not in input_json or "role" not in input_json["persona"]:
            raise ValueError("Missing required field: persona.role")

        if (
            "job_to_be_done" not in input_json
            or "task" not in input_json["job_to_be_done"]
        ):
            raise ValueError("Missing required field: job_to_be_done.task")

        # Display challenge info if available
        if "challenge_info" in input_json:
            challenge = input_json["challenge_info"]
            print(
                f"🏆 Challenge: {challenge.get('challenge_id', 'Unknown')} - {challenge.get('description', 'No description')}"
            )
            print(f"📋 Test Case: {challenge.get('test_case_name', 'Unknown')}")

        # Display document metadata if available
        if "documents" in input_json:
            print(f"📁 Found {len(input_json['documents'])} documents in metadata:")
            for doc in input_json["documents"][:3]:  # Show first 3
                print(
                    f"   • {doc.get('filename', 'Unknown')} - {doc.get('title', 'No title')}"
                )
            if len(input_json["documents"]) > 3:
                print(f"   ... and {len(input_json['documents']) - 3} more")

        return input_json

    except Exception as e:
        print(f"Error loading JSON input: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Intelligence with JSON Input Only')
    
    # Core arguments
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    # Input source - either directory or single PDF (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', help='Directory with PDF documents')
    input_group.add_argument('--pdf_path', help='Path to single PDF file')
    
    # JSON input OR persona/job (mutually exclusive)
    json_group = parser.add_mutually_exclusive_group(required=False)
    json_group.add_argument('--json_input', help='Path to input JSON file')
    
    # Alternative: direct persona/job input
    persona_group = json_group.add_argument_group()
    parser.add_argument('--persona', help='Persona description')
    parser.add_argument('--job', help='Job-to-be-done')
    
    args = parser.parse_args()
    
    # Validate persona/job combination
    if args.persona or args.job:
        if not (args.persona and args.job):
            print("❌ Error: Both --persona and --job are required when not using --json_input")
            exit(1)
    
    # Load or create input JSON
    if args.json_input:
        try:
            input_json = load_input_json(args.json_input)
            print(f"📋 Using provided input JSON: {args.json_input}")
        except Exception as e:
            print(f"❌ Error loading input JSON: {e}")
            exit(1)
    else:
        # Create input JSON from persona/job
        input_json = {
            "persona": {"role": args.persona},
            "job_to_be_done": {"task": args.job}
        }
        print(f"📋 Using direct persona/job input")

    # Determine input source
    if args.pdf_path:
        print(f"📄 Processing single PDF: {args.pdf_path}")
        input_source = args.pdf_path
        is_single_pdf = True
    else:
        print(f"📁 Processing directory: {args.input_dir}")
        input_source = args.input_dir
        is_single_pdf = False

    # Process the collection
    try:
        
        result = process_collection_from_json(input_json, input_source)

        # Save results
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 Processing complete!")
        print(f"📁 Results saved to: {args.output}")
        print(
            f"📊 Final results: {len(result['extracted_sections'])} sections, {len(result['subsection_analysis'])} subsections"
        )

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        exit(1)
