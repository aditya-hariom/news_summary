import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 11 * 72 - 30, "Minor Project 2026 — Contradiction-Aware Multi-Document Summarizer")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, 11 * 72 - 34, 8.5 * 72 - 40, 11 * 72 - 34)
            
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 40, 8.5 * 72 - 40, 40)
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 40, 28, page_str)
        self.drawString(40, 28, "CONFIDENTIAL & PROPRIETARY — ACADEMIC PROJECT REPORT")
        self.restoreState()


def generate_pdf(filename="Project_Implementation_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#0F172A")    # Deep Navy
    accent_blue = colors.HexColor("#1D4ED8")      # Royal Blue
    accent_crimson = colors.HexColor("#BE123C")   # Crimson Red
    accent_green = colors.HexColor("#15803D")     # Emerald Green
    body_text_color = colors.HexColor("#1E293B")  # Dark Slate

    styles.add(ParagraphStyle(
        name='DocTitle',
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
        textColor=primary_color,
        alignment=0,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        name='DocSubtitle',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=accent_blue,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='SubSectionHeading',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=accent_blue,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='BodyCustom',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=body_text_color,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name='BodyBold',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=body_text_color,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name='CalloutText',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    ))

    styles.add(ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=body_text_color
    ))

    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.white
    ))

    styles.add(ParagraphStyle(
        name='QText',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0F172A")
    ))

    styles.add(ParagraphStyle(
        name='AText',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    ))

    story = []

    # ==================== HEADER / HERO SECTION ====================
    hero_data = [
        [
            Paragraph("<b>CONTRADICTION-AWARE MULTI-DOCUMENT NEWS SUMMARIZER</b>", styles['DocTitle']),
            Paragraph("<b>MINOR PROJECT 2026</b><br/>Domain: NLP & AI<br/>Status: Fully Implemented", styles['CalloutText'])
        ],
        [
            Paragraph("Technical Architecture, Implementation Details & Oral Viva Defense Guide", styles['DocSubtitle']),
            Paragraph("Framework: Python / Groq / Streamlit", styles['CalloutText'])
        ]
    ]
    hero_table = Table(hero_data, colWidths=[4.8 * inch, 2.4 * inch])
    hero_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(hero_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F172A"), spaceAfter=10))

    # ==================== 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT ====================
    story.append(Paragraph("1. Executive Summary & Problem Statement", styles['SectionHeading']))
    p1 = (
        "During high-stakes breaking events (e.g., natural disasters, geopolitical crises, industrial accidents), "
        "multiple independent news agencies publish rapidly updating reports with conflicting statistics, dates, "
        "casualty numbers, and official attributions. "
        "Conventional Large Language Model (LLM) summarizers (such as ChatGPT or Gemini) generate single narrative summaries "
        "that suffer from <b>attention dilution</b> and <b>hallucinatory synthesis</b>—often fabricating synthetic averages "
        "(e.g., <i>'between 47 and 100 casualties were reported'</i>) while stripping away vital source attributions."
    )
    story.append(Paragraph(p1, styles['BodyCustom']))

    # Comparison Table (Standard AI vs Our System)
    comp_data = [
        [Paragraph("<b>Standard LLM Summarizer (ChatGPT / Gemini)</b>", styles['TableHeader']),
         Paragraph("<b>Our Contradiction-Aware System</b>", styles['TableHeader'])],
        [
            Paragraph("• Blends conflicting facts into a vague synthetic consensus.<br/>"
                      "• Loses source traceability (who said what).<br/>"
                      "• Hallucinates false middle-ground numbers.<br/>"
                      "• Disguises ground reporting discrepancies as verified fact.", styles['TableCell']),
            Paragraph("• Explicitly isolates factual contradictions.<br/>"
                      "• 100% source attribution preserved end-to-end.<br/>"
                      "• Places conflicting reports <b>side-by-side</b> (VS comparison).<br/>"
                      "• Classifies into Confirmed, Disputed, and Unconfirmed buckets.", styles['TableCell'])
        ]
    ]
    comp_table = Table(comp_data, colWidths=[3.6 * inch, 3.6 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#991B1B")),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#166534")),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor("#FFF1F2")),
        ('BACKGROUND', (1, 1), (1, 1), colors.HexColor("#F0FDF4")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 8))

    # ==================== 2. END-TO-END PIPELINE ARCHITECTURE ====================
    story.append(Paragraph("2. End-to-End System Pipeline & Workflow", styles['SectionHeading']))
    p2 = (
        "The system adopts a modular 4-stage pipeline that processes unstructured news documents into "
        "an attributed, contradiction-aware intelligence report:"
    )
    story.append(Paragraph(p2, styles['BodyCustom']))

    pipeline_table_data = [
        [
            Paragraph("<b>Stage</b>", styles['TableHeader']),
            Paragraph("<b>Module / File</b>", styles['TableHeader']),
            Paragraph("<b>Core Algorithm / Model</b>", styles['TableHeader']),
            Paragraph("<b>Input & Output Artifacts</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<b>Stage 1:<br/>Extraction</b>", styles['TableCell']),
            Paragraph("<code>src/01_extract_claims.py</code>", styles['TableCell']),
            Paragraph("Groq LLM (<code>qwen/qwen3.8-27b</code>)<br/>Prompt-engineered JSON extraction with numeric normalization", styles['TableCell']),
            Paragraph("<b>In:</b> Raw <code>.txt</code> news files<br/><b>Out:</b> <code>output/claims.json</code> (Atomic claims)", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage 2:<br/>Clustering</b>", styles['TableCell']),
            Paragraph("<code>src/02_cluster_and_classify.py</code>", styles['TableCell']),
            Paragraph("<code>all-MiniLM-L6-v2</code> embeddings + Agglomerative Clustering (Cosine metric, average linkage, Silhouette score auto-tuning)", styles['TableCell']),
            Paragraph("<b>In:</b> Atomic claims<br/><b>Out:</b> Semantic claim clusters grouping identical fact-slots", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage 3:<br/>Stance Detection</b>", styles['TableCell']),
            Paragraph("<code>src/02_cluster_and_classify.py</code>", styles['TableCell']),
            Paragraph("<b>Neuro-Symbolic Hybrid:</b><br/>• Symbolic: Disparity & temporal rules<br/>• Neural: Chain-of-Thought (CoT) FNC-1 Classifier", styles['TableCell']),
            Paragraph("<b>In:</b> Cross-source pairs in cluster<br/><b>Out:</b> <code>output/clusters_with_stance.json</code>", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage 4:<br/>Synthesis</b>", styles['TableCell']),
            Paragraph("<code>src/03_generate_summary.py</code>", styles['TableCell']),
            Paragraph("Attributed graph categorization into Confirmed, Disputed, and Unconfirmed buckets", styles['TableCell']),
            Paragraph("<b>In:</b> Classified stance edges<br/><b>Out:</b> <code>output/final_summary.md</code>", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage 5:<br/>Dashboard</b>", styles['TableCell']),
            Paragraph("<code>app.py</code> / <code>src/04_app.py</code>", styles['TableCell']),
            Paragraph("Interactive Web UI built in Streamlit with custom CSS & dynamic pipeline runner", styles['TableCell']),
            Paragraph("<b>Features:</b> Live file uploader, VS battle cards, live pipeline execution", styles['TableCell'])
        ]
    ]

    pipe_table = Table(pipeline_table_data, colWidths=[0.9 * inch, 1.6 * inch, 2.5 * inch, 2.2 * inch])
    pipe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(pipe_table)
    story.append(Spacer(1, 10))

    # ==================== 3. WHAT I IMPLEMENTED (MODULE DEEP DIVE) ====================
    story.append(Paragraph("3. Technical Implementation Details (What I Built)", styles['SectionHeading']))
    
    # Step 1 Deep Dive
    story.append(Paragraph("A. Atomic Claim Extraction Engine", styles['SubSectionHeading']))
    p_s1 = (
        "• <b>Compound Sentence Deconstruction:</b> News sentences often pack multiple statistics (e.g. <i>'82 died and 1.78 lakh were affected'</i>). "
        "The extractor breaks these into independent atomic claims so each fact can be clustered and verified on its own.<br/>"
        "• <b>Quantitative Normalization:</b> Automatically normalizes vernacular/regional metrics into standard numeric values "
        "(e.g., converting <code>'1.78 lakh'</code> ➔ <code>178000</code>, <code>'5 million'</code> ➔ <code>5000000</code>) to enable downstream arithmetic comparisons.<br/>"
        "• <b>Attribution Preservation:</b> Extracts metadata tags for each claim: <code>source_file</code>, <code>attribution</code> (e.g., Police, State Disaster Authority, Chief Minister), "
        "<code>date</code>, and <code>topic</code> category.<br/>"
        "• <b>Production-Grade Resilience:</b> Implements regex fallback parsing, JSON validation schemas, and exponential backoff retry logic to handle API rate limits."
    )
    story.append(Paragraph(p_s1, styles['BodyCustom']))

    # Step 2 Deep Dive
    story.append(Paragraph("B. Semantic Clustering with Adaptive Silhouette Optimization", styles['SubSectionHeading']))
    p_s2 = (
        "• <b>Dense Vector Embeddings:</b> Uses the <code>all-MiniLM-L6-v2</code> transformer model to convert claim text into 384-dimensional dense vectors.<br/>"
        "• <b>Boilerplate Filtering:</b> Strips generic event-level noise (such as <i>'in the 2026 floods'</i>) so that semantic clustering focuses purely on specific factual content.<br/>"
        "• <b>Agglomerative Hierarchical Clustering:</b> Employs cosine distance with average linkage. Unlike K-Means, it requires no predetermined number of clusters (K).<br/>"
        "• <b>Adaptive Silhouette Scoring:</b> Implemented <code>find_optimal_threshold()</code> which sweeps candidate distance thresholds (0.35 to 0.70) "
        "and automatically selects the threshold that maximizes the silhouette coefficient, ensuring optimal topical grouping."
    )
    story.append(Paragraph(p_s2, styles['BodyCustom']))

    # Step 3 Deep Dive
    story.append(Paragraph("C. Neuro-Symbolic Stance Classification (FNC-1 Standard)", styles['SubSectionHeading']))
    p_s3 = (
        "To achieve both high speed and deep contextual accuracy, a <b>hybrid neuro-symbolic architecture</b> was implemented:<br/>"
        "1. <b>Deterministic Symbolic Rules:</b><br/>"
        "   - <i>Numeric Disparity Rule:</i> If two sources report on the same event window but numeric statistics differ by &gt; 25% (e.g., 178,000 vs 720,000), "
        "it is immediately classified as <b>disagree</b> with 95% confidence.<br/>"
        "   - <i>Temporal Progression Rule:</i> If numbers differ across different dates, the system classifies them as <b>discuss</b> (temporal evolution, not contradiction).<br/>"
        "   - <i>Identical Match Rule:</i> Exact corroborated counts receive an automatic <b>agree</b> classification.<br/>"
        "2. <b>Neural Chain-of-Thought (CoT) Classifier:</b><br/>"
        "   - For qualitative and nuanced claims, Groq LLM evaluates the cross-source pair against the 4 FNC-1 stance categories: "
        "<code>agree</code>, <code>disagree</code>, <code>discuss</code>, or <code>unrelated</code>, outputting step-by-step reasoning and a confidence score."
    )
    story.append(Paragraph(p_s3, styles['BodyCustom']))

    # Step 4 Deep Dive
    story.append(Paragraph("D. Contradiction-Aware Report Synthesis", styles['SubSectionHeading']))
    p_s4 = (
        "The graph of claims and stance edges is partitioned into three distinct semantic tiers:<br/>"
        "• <b>🟢 Confirmed Facts:</b> Clusters supported by 2 or more independent outlets with active <code>agree</code> edges and zero disputes.<br/>"
        "• <b>🔴 Disputed Facts:</b> Fact clusters containing at least one verified <code>disagree</code> stance. "
        "<b>Core Design Mandate:</b> Conflicting claims are <i>never merged</i> into a single sentence. Instead, they are displayed side-by-side with source attribution and AI verification reasoning.<br/>"
        "• <b>⚪ Unconfirmed & Developing:</b> Claims reported by only one outlet or pending multi-source verification."
    )
    story.append(Paragraph(p_s4, styles['BodyCustom']))

    # Step 5 Deep Dive
    story.append(Paragraph("E. Interactive Web Application (Streamlit Dashboard)", styles['SubSectionHeading']))
    p_s5 = (
        "• <b>Modern UI & Aesthetics:</b> Custom CSS with dark hero header, Plus Jakarta Sans typography, and metric counters.<br/>"
        "• <b>VS Battle Cards:</b> Distinct red and blue cards highlighting conflicting headlines side-by-side for intuitive faculty demonstration.<br/>"
        "• <b>Live Multi-Document Upload:</b> Allows users to upload any 2+ news <code>.txt</code> files and run the entire 4-stage pipeline live.<br/>"
        "• <b>Downloadable Artifacts:</b> Direct download of the generated contradiction-aware Markdown intelligence report."
    )
    story.append(Paragraph(p_s5, styles['BodyCustom']))
    story.append(Spacer(1, 10))

    # ==================== 4. KEY RESULTS & VALIDATION ====================
    story.append(Paragraph("4. Experimental Validation & Test Datasets", styles['SectionHeading']))
    exp_text = (
        "The system was evaluated against two distinct test corpora to validate multi-domain adaptability:"
    )
    story.append(Paragraph(exp_text, styles['BodyCustom']))

    test_data = [
        [
            Paragraph("<b>Test Corpus</b>", styles['TableHeader']),
            Paragraph("<b>Sources Evaluated</b>", styles['TableHeader']),
            Paragraph("<b>Detected Factual Discrepancy</b>", styles['TableHeader']),
            Paragraph("<b>Pipeline Outcome</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<b>2026 Assam Floods</b><br/>(Disaster Domain)", styles['TableCell']),
            Paragraph("• Wikipedia (Compiled)<br/>• NPR International<br/>• Times of India<br/>• MapsOfIndia", styles['TableCell']),
            Paragraph("<b>People Affected:</b><br/>• TOI: 1.78 Lakh (178,000)<br/>• MapsOfIndia: 7.2 Lakh (720,000)<br/><i>(4x Disparity for same timeframe)</i>", styles['TableCell']),
            Paragraph("<b>Flagged as DISPUTE:</b><br/>Pair isolated side-by-side with 95% confidence and AI reasoning.", styles['TableCell'])
        ],
        [
            Paragraph("<b>New Delhi Protests</b><br/>(Political Domain)", styles['TableCell']),
            Paragraph("• Al Jazeera Report<br/>• Delhi Police Official", styles['TableCell']),
            Paragraph("<b>Casualty Figures:</b><br/>• Al Jazeera: ~150 protesters hurt<br/>• Police: 180 total (60 protesters + 118 police)", styles['TableCell']),
            Paragraph("<b>Corroborated / Categorized:</b><br/>Single-source political statements isolated to Unconfirmed tier.", styles['TableCell'])
        ]
    ]
    test_table = Table(test_data, colWidths=[1.5 * inch, 1.8 * inch, 2.3 * inch, 1.6 * inch])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 10))

    # ==================== 5. VIVA VOCE & ORAL DEFENSE PREPARATION ====================
    story.append(Paragraph("5. Oral Defense & Viva Voce Q&A (Frequently Asked Questions)", styles['SectionHeading']))

    viva_qa = [
        (
            "Q1: Why not simply feed all documents to ChatGPT and ask it to find contradictions?",
            "Standard LLMs suffer from 'Lost-in-the-Middle' attention degradation and inherent pressure to produce fluent, unified prose. When presented with conflicting numbers, they synthesize artificial compromise averages (e.g. 'between 47 and 100 casualties occurred') and discard the specific source attributions. Our modular pipeline guarantees mathematical clustering and deterministic checks before LLM verification, ensuring zero loss of provenance."
        ),
        (
            "Q2: What is an 'atomic claim' and why is sentence deconstruction necessary?",
            "An atomic claim is an indivisible, self-contained proposition that can be evaluated as true or false independently. Journalists regularly combine multiple metrics into one sentence (e.g. '82 people died and 1.78 lakh were displaced across 15 districts'). If we clustered entire compound sentences, vectors would represent noisy mixtures of topics. Deconstruction into atomic claims ensures that casualty stats are clustered strictly with casualty stats, and displacement stats strictly with displacement stats."
        ),
        (
            "Q3: Why Agglomerative Hierarchical Clustering instead of K-Means or DBSCAN?",
            "In dynamic multi-document news streams, the number of factual topics (K) is completely unknown beforehand, rendering K-Means impractical. DBSCAN requires rigid density estimation that struggles with small, variable-sized claim sets. Agglomerative Clustering builds a hierarchical distance tree using cosine similarity, and our automated Silhouette Scoring sweeps candidate thresholds to objectively select the optimal cluster boundary."
        ),
        (
            "Q4: If two reports give different numbers on different dates, is that a contradiction?",
            "No. That represents chronological or temporal progression, not factual contradiction. In disaster reporting, a death toll naturally rises from 66 on July 26 to 82 on August 2 and 98 on August 8. Our Temporal Progression Rule explicitly checks dates: if dates differ and numbers show logical advancement, the relationship is labeled 'discuss' rather than 'disagree'."
        ),
        (
            "Q5: What are the primary practical applications of this system?",
            "1. Emergency Disaster Relief: Coordinating real-time relief when municipal and NGO tallies conflict.\n2. Fact-Checking Newsrooms (Snopes, AltNews, Reuters Fact Check): Rapidly detecting media inflation and misreporting.\n3. Financial & Legal Due Diligence: Comparing conflicting financial statements and analyst forecasts during corporate mergers."
        )
    ]

    for q, a in viva_qa:
        q_box = [
            [Paragraph(f"<b>{q}</b>", styles['QText'])],
            [Paragraph(a.replace("\n", "<br/>"), styles['AText'])]
        ]
        t = Table(q_box, colWidths=[7.2 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    # ==================== 6. CONCLUSION & TECH STACK SUMMARY ====================
    story.append(Spacer(1, 4))
    story.append(Paragraph("6. Project Summary & Technical Stack Specification", styles['SectionHeading']))
    summary_box_data = [
        [
            Paragraph("<b>Component</b>", styles['TableHeader']),
            Paragraph("<b>Technology</b>", styles['TableHeader']),
            Paragraph("<b>Purpose in Implementation</b>", styles['TableHeader'])
        ],
        [Paragraph("Language", styles['TableCell']), Paragraph("Python 3.10+ / 3.14", styles['TableCell']), Paragraph("Core software implementation", styles['TableCell'])],
        [Paragraph("LLM Engine", styles['TableCell']), Paragraph("Groq Cloud API (Qwen 3.8 27B)", styles['TableCell']), Paragraph("Ultra-low-latency atomic extraction & CoT stance", styles['TableCell'])],
        [Paragraph("Embeddings", styles['TableCell']), Paragraph("SentenceTransformers (MiniLM-L6-v2)", styles['TableCell']), Paragraph("384-dimensional dense semantic representations", styles['TableCell'])],
        [Paragraph("Clustering", styles['TableCell']), Paragraph("Scikit-Learn (Agglomerative, Silhouette)", styles['TableCell']), Paragraph("Adaptive cosine hierarchical fact grouping", styles['TableCell'])],
        [Paragraph("Stance Framework", styles['TableCell']), Paragraph("FNC-1 Benchmark Standard", styles['TableCell']), Paragraph("Agree / Disagree / Discuss / Unrelated classification", styles['TableCell'])],
        [Paragraph("Frontend UI", styles['TableCell']), Paragraph("Streamlit + Custom Modern CSS", styles['TableCell']), Paragraph("Interactive faculty dashboard, live demo & export", styles['TableCell'])],
    ]
    sum_table = Table(summary_box_data, colWidths=[1.5 * inch, 2.3 * inch, 3.4 * inch])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sum_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Successfully generated: {filename}")


if __name__ == "__main__":
    out_file = "Project_Implementation_Report.pdf"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    generate_pdf(out_file)
