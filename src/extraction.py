"""
Atomic Claim Extraction Module
-------------------------------
Extracts atomic, verifiable factual claims from news documents using Groq LLM.
Preserves source attribution, normalizes quantitative data, and structures
claims with unique identifiers and attributes.
"""

import os
import re
import json
import glob
import time
from typing import List, Dict, Any, Optional
from groq import Groq

from src.config import (
    get_groq_api_key,
    GROQ_MODEL,
    EXTRACTION_MAX_TOKENS,
    RAW_SOURCES_DIR,
    CLAIMS_FILE,
)

EXTRACTION_SYSTEM_PROMPT = """You are a precision factual claim extraction engine for news analysis across any domain (disasters, accidents, politics, finance, health).
Your job is to read news reports and extract every ATOMIC FACTUAL CLAIM.
An atomic claim is a single, self-contained factual statement that can be verified independently.
Do NOT extract opinions, speculation, emotional reactions, or vague metaphors. Only extract verifiable factual claims.
Output MUST be valid JSON only."""

EXTRACTION_USER_PROMPT = """Read the source document below and extract all ATOMIC FACTUAL CLAIMS.

Guidelines:
1. Break compound sentences into separate, atomic claims.
   Example: "A bridge collapsed on Friday, killing 5 people and injuring 20."
   -> Claim 1: "A bridge collapsed on Friday."
   -> Claim 2: "5 people were killed in the bridge collapse."
   -> Claim 3: "20 people were injured in the bridge collapse."
2. For any claim containing a quantitative figure, extract the normalized numerical_value (convert '1.78 lakh' -> 178000, '7.2 lakh' -> 720000, '5 million' -> 5000000, '52' -> 52).
3. Assign a concise topic category ('casualties', 'affected_population', 'infrastructure_damage', 'financial_loss', 'rescue_relief', 'status_event', 'official_statement').
4. Extract structured semantic attributes:
   - subject (e.g. 'flood death toll', 'Kavery River', 'bridge')
   - predicate (e.g. 'reached', 'overflowed', 'collapsed')
   - numerical_value (number or null)
   - numerical_unit (e.g. 'deaths', 'injured', 'districts', 'people', or null)
5. Preserve explicit reporting date mentioned in or for the claim, or null.
6. Preserve attribution (who stated it: e.g. 'District Disaster Management Office', 'Times of India', 'police', 'officials', or null).
7. Output valid JSON with a single top-level key named "claims".

Few-Shot Examples:

Example A (Disaster Report):
Source: "On 2 August, Times of India reported that 1.78 lakh people were affected across 15 districts, with 82 dead."
Output:
{{
  "claims": [
    {{
      "claim": "1.78 lakh people were affected across 15 districts.",
      "normalized_claim": "178000 people were affected across 15 districts.",
      "topic": "affected_population",
      "subject": "affected population",
      "predicate": "affected across districts",
      "numerical_value": 178000,
      "numerical_unit": "people",
      "date": "2 August 2026",
      "attribution": "Times of India"
    }},
    {{
      "claim": "The death toll climbed to 82.",
      "normalized_claim": "82 people died in the incident.",
      "topic": "casualties",
      "subject": "death toll",
      "predicate": "climbed to",
      "numerical_value": 82,
      "numerical_unit": "deaths",
      "date": "2 August 2026",
      "attribution": "Times of India"
    }}
  ]
}}

Example B (Accident / Infrastructure):
Source: "A bridge collapsed near Karnal on Friday morning. Five people were injured. Rescue teams reached the location. Officials have not confirmed the cause."
Output:
{{
  "claims": [
    {{
      "claim": "A bridge collapsed near Karnal on Friday morning.",
      "normalized_claim": "A bridge collapsed near Karnal on Friday morning.",
      "topic": "status_event",
      "subject": "bridge",
      "predicate": "collapsed",
      "numerical_value": null,
      "numerical_unit": null,
      "date": "Friday morning",
      "attribution": null
    }},
    {{
      "claim": "Five people were injured in the collapse.",
      "normalized_claim": "5 people were injured.",
      "topic": "casualties",
      "subject": "people",
      "predicate": "injured",
      "numerical_value": 5,
      "numerical_unit": "injured",
      "date": "Friday morning",
      "attribution": null
    }},
    {{
      "claim": "Rescue teams reached the collapse location.",
      "normalized_claim": "Rescue teams reached the collapse site.",
      "topic": "rescue_relief",
      "subject": "rescue teams",
      "predicate": "reached location",
      "numerical_value": null,
      "numerical_unit": null,
      "date": null,
      "attribution": null
    }},
    {{
      "claim": "Officials have not confirmed the cause of the collapse.",
      "normalized_claim": "The cause of the bridge collapse remains unconfirmed by officials.",
      "topic": "official_statement",
      "subject": "cause of collapse",
      "predicate": "not confirmed",
      "numerical_value": null,
      "numerical_unit": null,
      "date": null,
      "attribution": "Officials"
    }}
  ]
}}

Now extract all atomic claims from this document:

SOURCE DOCUMENT:
{document_text}
"""


def parse_claims_json(raw_text: str) -> List[Dict[str, Any]]:
    """
    Safely extracts and parses JSON claims from raw LLM output.
    Handles markdown fences, extraneous prose, and structure variances.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Direct JSON parse attempt
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "claims" in data and isinstance(data["claims"], list):
                return data["claims"]
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Regex extraction fallback for object pattern: { "claims": [...] }
    match_dict = re.search(r"\{\s*\"claims\"\s*:\s*\[.*?\]\s*\}", cleaned, re.DOTALL)
    if match_dict:
        try:
            data = json.loads(match_dict.group(0))
            return data.get("claims", [])
        except json.JSONDecodeError:
            pass

    # Regex extraction fallback for list pattern: [ {...}, {...} ]
    match_list = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if match_list:
        try:
            data = json.loads(match_list.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse claims JSON from output: {raw_text[:200]}...")


def validate_and_normalize_claim(
    raw_claim: Dict[str, Any],
    source_name: str,
    article_id: str,
    index: int
) -> Dict[str, Any]:
    """
    Validates, standardizes, and enriches a single claim dictionary with IDs,
    normalized values, and backward-compatible fields.
    """
    claim_text = str(raw_claim.get("claim", "")).strip()
    if not claim_text:
        claim_text = str(raw_claim.get("extracted_claim", "")).strip()

    normalized_text = str(raw_claim.get("normalized_claim") or claim_text).strip()
    topic = str(raw_claim.get("topic") or raw_claim.get("category") or "general").strip().lower()

    # Numerical parsing with safety checks
    num_val = raw_claim.get("numerical_value")
    if num_val is not None:
        try:
            if isinstance(num_val, str):
                num_cleaned = re.sub(r"[^\d.]", "", num_val)
                num_val = float(num_cleaned) if "." in num_cleaned else int(num_cleaned)
            elif isinstance(num_val, (int, float)):
                num_val = int(num_val) if float(num_val).is_integer() else float(num_val)
        except (ValueError, TypeError):
            num_val = None

    num_unit = raw_claim.get("numerical_unit")
    if num_unit:
        num_unit = str(num_unit).strip().lower()

    date_str = raw_claim.get("date")
    if date_str:
        date_str = str(date_str).strip()

    attribution = raw_claim.get("attribution")
    if attribution:
        attribution = str(attribution).strip()

    claim_id = f"clm_{article_id}_{index:02d}"

    subject = str(raw_claim.get("subject") or topic)
    predicate = str(raw_claim.get("predicate") or "reported")

    structured_attributes = {
        "subject": subject,
        "predicate": predicate,
        "numerical_value": num_val,
        "numerical_unit": num_unit,
        "location": raw_claim.get("location"),
        "temporal_anchor": date_str,
    }

    # Unified schema (both modern and legacy keys for complete backward compatibility)
    return {
        "claim_id": claim_id,
        "article_id": article_id,
        "source_file": source_name,         # Legacy key
        "source_name": source_name,         # Modern key
        "claim": claim_text,                # Legacy key
        "extracted_claim": claim_text,      # Modern key
        "normalized_claim": normalized_text,
        "topic": topic,                     # Legacy key
        "category": topic,                  # Modern key
        "numerical_value": num_val,
        "numerical_unit": num_unit,
        "date": date_str,
        "attribution": attribution,
        "structured_attributes": structured_attributes,
        "confidence": float(raw_claim.get("confidence", 0.95)),
    }


def extract_claims_from_text(
    text: str,
    source_name: str = "custom_article",
    article_id: Optional[str] = None,
    client: Optional[Groq] = None,
    max_retries: int = 4
) -> List[Dict[str, Any]]:
    """
    Extracts atomic claims from raw article text using Groq LLM.
    Supports both file-based and pasted text workflows.
    """
    if client is None:
        api_key = get_groq_api_key()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it in .env or system environment.")
        client = Groq(api_key=api_key)

    if not article_id:
        article_id = re.sub(r"[^\w-]", "_", source_name).lower()

    prompt_content = EXTRACTION_USER_PROMPT.format(document_text=text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=EXTRACTION_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            raw_claims = parse_claims_json(raw)

            if not raw_claims:
                raise ValueError("Parsed claims list is empty.")

            validated_claims = []
            for idx, c in enumerate(raw_claims, 1):
                validated = validate_and_normalize_claim(c, source_name, article_id, idx)
                if validated["claim"]:
                    validated_claims.append(validated)

            return validated_claims

        except Exception as e:
            err_msg = str(e)
            print(f"  [Attempt {attempt}/{max_retries}] Extraction notice for '{source_name}': {err_msg[:100]}")
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "tokens" in err_msg.lower():
                wait_time = 3.0 * attempt
                print(f"  [Rate limit wait] Sleeping {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries:
                time.sleep(1.5)
            else:
                print(f"[ERROR] Failed to extract claims for '{source_name}' after {max_retries} attempts.")
                raise


def extract_claims_from_file(filepath: str, client: Optional[Groq] = None, max_retries: int = 4) -> List[Dict[str, Any]]:
    """
    Reads a single .txt news file and extracts atomic claims.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    source_name = os.path.splitext(os.path.basename(filepath))[0]
    return extract_claims_from_text(
        text=text,
        source_name=source_name,
        article_id=source_name,
        client=client,
        max_retries=max_retries
    )


def run_extraction_pipeline(input_files: Optional[List[str]] = None, output_path: str = CLAIMS_FILE) -> List[Dict[str, Any]]:
    """
    Executes the batch extraction pipeline across source files and writes output/claims.json.
    """
    if input_files is None:
        all_candidates = sorted(glob.glob(os.path.join(RAW_SOURCES_DIR, "*.txt")))
        input_files = [f for f in all_candidates if not os.path.basename(f).startswith("README")]

    if not input_files:
        raise FileNotFoundError(f"No source files found in {RAW_SOURCES_DIR} to process.")

    print(f"\n=======================================================")
    print(f"STAGE 1: ATOMIC CLAIM EXTRACTION ({len(input_files)} documents)")
    print(f"=======================================================")

    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
    client = Groq(api_key=api_key)

    all_claims = []
    for filepath in input_files:
        filename = os.path.basename(filepath)
        print(f"Extracting claims from: {filename} ...")
        claims = extract_claims_from_file(filepath, client=client)
        print(f"  -> Extracted {len(claims)} atomic claims")
        all_claims.extend(claims)
        time.sleep(2.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Successfully saved {len(all_claims)} atomic claims to {output_path}")
    return all_claims


if __name__ == "__main__":
    run_extraction_pipeline()
