# College Minor Project Viva Preparation Guide
## Contradiction-Aware Multi-Document News Analysis Using Large Language Models

This comprehensive viva guide explains every technical concept, architectural decision, and potential examiner question in **simple Hinglish** (for easy understanding) and **formal English** (to answer during your college viva).

---

## 📌 Section 1: Project Overview & Core Motivation

### Q1. What is the core problem your project solves?
* **Simple Hinglish Answer:**
  > *"Sir, jab koi breaking news (jaise flood, disaster ya accident) aati hai, to alag-alag media outlets alag-alag reporting karte hain. For example, ek newspaper bolta hai 47 log mare, doosra bolta hai 82 log mare, aur teesra bolta hai 100 log mare.
  > Traditional summarizers (jaise ChatGPT, Gemini, ya T5) in conflicting facts ko mix karke ek vague narrative bana dete hain (jaise: 'Between 47 to 100 people died'). Isse reporting errors chhup jaate hain aur hallucination hoti hai.
  > Hamara system facts ko blend karne ke bajaye conflicting claims ko side-by-side exact source name aur timestamp ke sath highlight karta hai taaki reader ko 100% transparency mile."*
* **Technical English Answer:**
  > *"Conventional multi-document summarization models suffer from consensus hallucination and information loss—they synthesize conflicting factual statements into an artificial compromise. Our system is contradiction-aware: it extracts atomic claims, clusters them by topical semantics, performs neuro-symbolic stance classification, and presents disagreements side-by-side with full source attribution."*

### Q2. What is the novelty of your project?
* **Viva Key Points:**
  1. **Source Attribution Preservation:** No claim is ever separated from its reporting source.
  2. **Epistemic Modesty:** The system never declares a claim as "absolute objective truth" just because 3 out of 4 outlets repeated it.
  3. **Temporal Progression vs. True Contradiction:** Distinguishes between figures evolving naturally over dates vs actual incompatible factual clashes.
  4. **Neuro-Symbolic Architecture:** Combines deterministic mathematical rules (for numbers and units) with LLM Chain-of-Thought (for contextual intent).

---

## 🔬 Section 2: Technical Deep-Dive

### Q3. What is an "Atomic Claim" and why do we need it?
* **Simple Hinglish Answer:**
  > *"Compound sentences me ek sath multiple facts hote hain (jaise: 'A bridge collapsed near Karnal, injuring five people and stalling traffic'). Agar hum is poore sentence ko doosre article ke sentence se compare karenge, to comparison fail ho jayega kyunki ek part match ho sakta hai aur doosra nahi.
  > Atomic Claim ek single, self-contained statement hota hai jisme sirf ek verifiable fact hota hai (e.g., Claim 1: Bridge collapsed, Claim 2: Five people injured). Granular claims banana clustering aur contradiction detection ko 100% accurate banata hai."*
* **Technical English Answer:**
  > *"An atomic claim is a minimal, independently verifiable proposition containing a single subject-predicate assertion. Decomposing unstructured multi-sentence articles into atomic units prevents semantic entanglement and enables precise pairwise alignment across disparate documents."*

### Q4. Why did you use Sentence-Transformers (`all-MiniLM-L6-v2`) instead of traditional TF-IDF or Word2Vec?
* **Simple Hinglish Answer:**
  > *"TF-IDF sirf exact word matches dekhta hai. Agar ek outlet '50 dead' likhe aur doosra 'half a century fatalities' likhe, to TF-IDF unhe match nahi kar payega kyunki koi common word nahi hai.
  > `all-MiniLM-L6-v2` ek dense embedding model hai jo 384-dimensional vector space me semantic meaning (arth) ko capture karta hai, chahe vocabulary completely different ho."*
* **Technical English Answer:**
  > *"Dense bi-encoder architectures like all-MiniLM-L6-v2 map sentences to a shared semantic latent space using cosine distance. Unlike sparse lexical representations (TF-IDF/BM25), bi-encoders identify topical equivalence despite radical lexical divergence and syntactic variation."*

### Q5. Why Agglomerative Hierarchical Clustering instead of K-Means?
* **Simple Hinglish Answer:**
  > *"K-Means algorithm me hume pehle se 'K' (number of clusters) specify karna padta hai. Breaking news me hume pehle se kaise pata chalega ki kitne factual topics hain?
  > Agglomerative Clustering ek bottom-up hierarchical method hai. Isme hum distance threshold set karte hain aur model naturally similar facts ko group karta hai bina pre-defined K ke."*
* **Technical English Answer:**
  > *"K-Means requires an a-priori specification of the number of centroids (K) and assumes spherical cluster geometry. News corpora feature unpredictable topic counts. Agglomerative Hierarchical Clustering with average linkage and cosine distance allows natural topic emergence bounded by a distance threshold, optimized adaptively via Silhouette scoring."*

### Q6. What is the Silhouette Score and how does your system use it?
* **Explanation:**
  > *"Silhouette score measures how similar an object is to its own cluster compared to other clusters (ranging from -1 to +1). Our system automatically evaluates candidate distance thresholds [0.35, 0.45, 0.50, 0.55, 0.60, 0.70] and selects the threshold that maximizes cluster cohesion and separation."*

### Q7. Why a Neuro-Symbolic Hybrid approach instead of pure LLM prompting?
* **Simple Hinglish Answer:**
  > *"Agar hum har single pair ko LLM ke paas bhejenge, to do problems aati hain:
  > 1. LLMs exact math aur ratio comparisons me hallucinate kar sakte hain ya inconsistent hote hain.
  > 2. API calls slow hoti hain aur token rate limits (OTPM) trigger karti hain.
  > Isliye humne Neuro-Symbolic approach banayi: Pehle symbolic rules (Python math) exact identical text, identical numbers, unit mismatches, aur date progression ko instantaneous 0-second execution me solve kar dete hain. Sirf nuanced language intent ke liye LLM (Neural) call hota hai."*

### Q8. What is the Stance Classification Scheme (FNC-1)?
* **5 Explicit Classes:**
  1. **`AGREE`**: Both sources support the exact same fact (e.g. 52 deaths confirmed by Dainik Samachar & EastWind News).
  2. **`DISAGREE`**: Incompatible claims for the same event window (e.g. 1.78 Lakh vs 7.2 Lakh affected).
  3. **`DISCUSS`**: Chronological progression across distinct dates (e.g. 66 deaths on July 26 $\to$ 100 deaths on August 10).
  4. **`UNRELATED`**: Different metrics or topics that landed in the same cluster (e.g. deaths vs food packets).
  5. **`INSUFFICIENT_EVIDENCE`**: Ambiguous qualifiers or missing temporal/geographical anchors.

---

## 📊 Section 3: Evaluation & Experimental Results

### Q9. How did you evaluate your system?
* **Viva Defense Answer:**
  > *"We evaluated our stance engine against a manually annotated 20-pair ground-truth benchmark (`data/evaluation_benchmark/benchmark_pairs.json`) covering disasters, railway crashes, layoffs, and aviation incidents.
  > Our system achieved **75.0% Overall Accuracy** and a **Macro F1-Score of 0.600** (with 1.00 F1 on `AGREE` and 0.83 F1 on `DISAGREE`)."

### Q10. What errors did you observe during evaluation? (Examiner's Favorite Question!)
* **Honest Error Analysis (Demonstrates True Mastery):**
  1. **Official Revision vs. Temporal Progression:** In the Balasore train derailment benchmark (Pair 03), 288 deaths were reported on June 3 and revised down to 275 deaths on June 4. The model classified this as `DISCUSS` (assuming natural date progression) rather than `DISAGREE` (downward official revision).
  2. **Qualitative Ambiguity:** In Pair 15, "heavy damages" vs "3 houses cracked" was classified as `DISAGREE` by the model due to scale disparity, whereas the ground-truth was `INSUFFICIENT_EVIDENCE` due to lack of shared geographical scope.

---

## ⚠️ Section 4: System Limitations & Future Work

### Limitations of Current Work:
1. **Single-Sentence Claim Granularity:** Complex multi-clause context across multiple paragraphs can occasionally miss antecedent pronouns.
2. **Rate Limit Sensitivity on Free Tiers:** On-demand API tiers enforce strict token ceilings per minute, requiring exponential backoff pacing.
3. **English-Only Evaluation:** The current pipeline focuses on English news reports; regional Indian languages (Hindi, Assamese) require cross-lingual embedding models (e.g., `paraphrase-multilingual-mpnet-base-v2`).

### Future Scope:
1. Automated RSS and live news scraping connectors.
2. Cross-lingual contradiction detection (e.g. comparing Hindi local reports vs English wire agencies).
3. Automated chronological timeline reconstruction with Sankey diagrams.

---

## 🎯 Section 5: Step-by-Step Viva Demo Script

When the examiner says: *"Show me the project running."*

1. **Step 1:** Open terminal and run:
   ```bash
   streamlit run app.py
   ```
2. **Step 2:** Point out the **Hero Metrics**:
   - *"Sir, as you can see, the system has analyzed 4 independent news agencies covering the Assam floods, extracted 19 atomic claims, and grouped them into 3 semantic fact clusters."*
3. **Step 3:** Click on **Tab 1: Contradiction Explorer**:
   - Show the side-by-side **VS card**: *"Notice here: MapsOfIndia claims 7.2 Lakh affected, while Times of India reports 1.78 Lakh. The system calculates a 4.0x disparity and highlights both with exact attribution without taking a side."*
4. **Step 4:** Click on **Tab 3: Claims Explorer**:
   - Type `"death"` or filter by outlet to demonstrate dynamic search and structured attributes (`numerical_value`, `unit`, `attribution`).
5. **Step 5:** Show **Tab 7: Run Live Analysis**:
   - Paste two short sentences into the custom text box (e.g., *Outlet A: 5 people injured* vs *Outlet B: 8 people injured*) and click Run to demonstrate live dynamic analysis.
