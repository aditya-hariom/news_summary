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
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = str(st.secrets["GROQ_API_KEY"]).strip()
            os.environ["GROQ_API_KEY"] = api_key
    except Exception:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please ensure it is present in your .env file or Streamlit Cloud Secrets.")

client = Groq(api_key=api_key)

MODEL_NAME = "qwen/qwen3.8-27b"

EXTRACTION_PROMPT = """You are an advanced factual claim extraction engine for news analysis across any domain (disasters, accidents, finance, elections, conflicts, health).

Read the source document below and extract every ATOMIC FACTUAL CLAIM it contains.
An atomic claim is a single, self-contained factual statement (specifically concerning counts, metrics, events, status, damage, casualties, or official statements) that can be verified independently.

Rules:
1. Break compound sentences into separate, self-contained atomic claims.
2. For any claim containing a quantitative metric, extract the normalized numerical_value (convert '1.78 lakh' -> 178000, '7.2 lakh' -> 720000, '5 million' -> 5000000, '47' -> 47).
3. Assign a concise 'topic' category (e.g., 'casualties', 'affected_population', 'damage', 'financial', 'rescue', 'status', 'statement').
4. Preserve explicit dates mentioned for the claim, or null.
5. Preserve attribution (who stated it: e.g. 'Times of India', 'officials', 'police', 'CEO', or null).
6. Output ONLY valid JSON with a top-level list named "claims".

Few-Shot Examples:
Example A (Disaster / Incident):
Source: "On 2 August, Times of India reported that 1.78 lakh people were affected across 15 districts, with 82 dead."
Output:
{{
  "claims": [
    {{
      "claim": "1.78 lakh people were affected across 15 districts.",
      "topic": "affected_population",
      "numerical_value": 178000,
      "numerical_unit": "people",
      "date": "2 August 2026",
      "attribution": "Times of India"
    }},
    {{
      "claim": "The death toll climbed to 82.",
      "topic": "casualties",
      "numerical_value": 82,
      "numerical_unit": "deaths",
      "date": "2 August 2026",
      "attribution": "Times of India"
    }}
  ]
}}

Example B (Business / General News):
Source: "On Monday, TechCorp announced 1,200 layoffs, though union leaders claim over 3,000 workers were terminated."
Output:
{{
  "claims": [
    {{
      "claim": "TechCorp announced 1,200 layoffs.",
      "topic": "layoffs",
      "numerical_value": 1200,
      "numerical_unit": "employees",
      "date": "Monday",
      "attribution": "TechCorp"
    }},
    {{
      "claim": "Union leaders claim over 3,000 workers were terminated.",
      "topic": "layoffs",
      "numerical_value": 3000,
      "numerical_unit": "employees",
      "date": "Monday",
      "attribution": "union leaders"
    }}
  ]
}}

Now extract all atomic claims from this document:

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


def extract_claims_from_file(filepath: str, max_retries: int = 5) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    source_name = os.path.splitext(os.path.basename(filepath))[0]
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=350,
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

            # Attach source_file and normalize schema fields
            for c in claims:
                c["source_file"] = source_name
                c["topic"] = c.get("topic", "general")
                c["numerical_value"] = c.get("numerical_value", None)
                c["numerical_unit"] = c.get("numerical_unit", None)
                c["attribution"] = c.get("attribution", None)
                c["date"] = c.get("date", None)

            return claims

        except Exception as e:
            err_msg = str(e)
            print(f"  [Attempt {attempt}/{max_retries}] Error processing {filepath}: {err_msg[:120]}")
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "tokens" in err_msg.lower():
                wait_time = 4.0 * attempt
                print(f"  [Rate limit pause] Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries:
                time.sleep(1.5)
            else:
                print(f"[ERROR] Could not extract claims for {filepath} after {max_retries} attempts.")
                raise


def main(input_files=None):
    if input_files is None:
        all_candidates = sorted(glob.glob("data/raw_sources/*.txt"))
        input_files = [f for f in all_candidates if not os.path.basename(f).startswith("README")]

    if not input_files:
        raise FileNotFoundError("No source files found in data/raw_sources/ to process.")

    print(f"Found {len(input_files)} source files to process.")
    all_claims = []

    for filepath in input_files:
        print(f"Extracting claims from {filepath} ...")
        claims = extract_claims_from_file(filepath)
        print(f"  -> {len(claims)} atomic claims extracted")
        all_claims.extend(claims)
        time.sleep(1.0)

    os.makedirs("output", exist_ok=True)
    output_path = "output/claims.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_claims)} total claims saved to {output_path}")


if __name__ == "__main__":
    main()
