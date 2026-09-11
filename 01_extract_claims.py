"""
Step 1: Claim Extraction Module
--------------------------------
Reads every .txt file in data/raw_sources/, sends each one to Groq LLM,
and extracts atomic factual claims as structured JSON.

Run: python src/01_extract_claims.py
Output: output/claims.json
"""

import os
import re
import json
import glob
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Try finding .env in workspace root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please ensure it is present in your .env file.")

client = Groq(api_key=api_key)

MODEL_NAME = "qwen/qwen3.8-27b"

EXTRACTION_PROMPT = """You are a factual claim extraction system for a news-analysis pipeline.

Read the source document below and extract every ATOMIC FACTUAL CLAIM it contains.
An atomic claim is a single, self-contained factual statement (usually about a number,
an event, a status, damage, casualties, or an attributed statement) that could be TRUE or FALSE independently
of other claims in the text.

Rules:
- Break compound sentences into separate atomic claims.
- Preserve any attribution (who stated it: "officials", "Times of India", "Chief Minister", "ASDMA", etc., or null if unattributed).
- Preserve any date mentioned for the claim (e.g. "26 July", "2 August", "10 August", or null if not stated).
- Do NOT include opinions, speculative commentary, or editorial background — only checkable factual claims.
- Output ONLY valid JSON containing a top-level list named "claims".

Output JSON format:
{{
  "claims": [
    {{
      "claim": "short atomic factual statement",
      "attribution": "who stated it, or null",
      "date": "date if mentioned, or null"
    }}
  ]
}}

SOURCE DOCUMENT:
{document_text}
"""


def parse_claims_json(raw_text: str) -> list:
    """Safely extracts and parses JSON claims from raw LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

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

    match_dict = re.search(r"\{\s*\"claims\"\s*:\s*\[.*?\]\s*\}", cleaned, re.DOTALL)
    if match_dict:
        try:
            data = json.loads(match_dict.group(0))
            return data.get("claims", [])
        except json.JSONDecodeError:
            pass

    match_list = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if match_list:
        try:
            data = json.loads(match_list.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse claims JSON from output: {raw_text[:200]}...")


def extract_claims_from_file(filepath: str, max_retries: int = 3) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    source_name = os.path.splitext(os.path.basename(filepath))[0]
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert news analyst that extracts factual claims into valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(document_text=text)
                    }
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            claims = parse_claims_json(raw)

            if not claims:
                raise ValueError("Parsed claims list is empty.")

            # Attach source_file to each claim
            for c in claims:
                c["source_file"] = source_name

            return claims

        except Exception as e:
            print(f"  [Attempt {attempt}/{max_retries}] Error processing {filepath}: {e}")
            if attempt < max_retries:
                time.sleep(1.5)
            else:
                print(f"[ERROR] Could not extract claims for {filepath} after {max_retries} attempts.")
                raise


def main():
    input_files = sorted(glob.glob("data/raw_sources/source*.txt"))
    if not input_files:
        raise FileNotFoundError("No source files found in data/raw_sources/ matching 'source*.txt'")

    print(f"Found {len(input_files)} source files to process.")
    all_claims = []

    for filepath in input_files:
        print(f"Extracting claims from {filepath} ...")
        claims = extract_claims_from_file(filepath)
        print(f"  -> {len(claims)} atomic claims extracted")
        all_claims.extend(claims)

    os.makedirs("output", exist_ok=True)
    output_path = "output/claims.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_claims)} total claims saved to {output_path}")


if __name__ == "__main__":
    main()
