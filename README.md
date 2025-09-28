# MARS Medical Summarization

**MARS Datathon 2025 – Medical Note Summarization & ICD-10 Code Prediction**  
This repository implements our competition pipeline, combining **LLM prompting with rule-based postprocessing** to achieve clinical accuracy, fairness, and format alignment in medical summarization tasks.

---

## Project Overview

### Task A. Brief Hospital Course Summarization
- **Input**: inpatient medical record (discharge summary)
- **Output**: 12–20 sentence narrative summary  
  *(structured: presentation → diagnostics/treatment → response → complications → disposition)*  
- **Strategy**: length-controlled prompting + cleanup of list/numbered outputs

### Task B. Radiology Impression Generation
- **Input**: Radiology Findings
- **Output**: `IMPRESSION:` header + 3–6 concise bullets
- **Strategy**: RSNA guidelines, explicit presence/absence phrasing, removal of management recommendations (`recommend`, `follow-up`, `consider`, etc.)

### Task C. ICD-10 Code Prediction
- **Input**: Discharge hospital course
- **Output**: up to 2 ICD-10 codes (uppercase, dot removed, deduplicated)  
- **Strategy**: regex extraction + ranking rules (non-R/Z preferred, specificity prioritized, overcoding suppressed)

---

## Key Features

- **Format Alignment**
  - A: match gold summary distribution (median ≈ 20 sentences)  
  - B: enforce `IMPRESSION:` + 3–6 bullets  
  - C: ≤2 codes per sample, matches dataset mean (1.58)

- **Rule-based Postprocessing**
  - Remove recommendation phrases → improves Conciseness & Clinical Clarity  
  - ICD normalization: uppercase, no dots, deduplication  

- **Fairness Guard**
  - Prepend `Gender` and `Age` metadata line to inputs  
  - Reduces subgroup performance disparity (80% rule monitoring)  

---

## Code Structure

```

├── submit.py           # Best submission (39.57) – Exaone + postprocessing
├── submit4.py          # Long-summary + header/bullet alignment
├── submit5.py          # Concise baseline-style version
├── processor.py        # Base Processor class (competition provided)
├── figures/            # Data distribution analysis & visualization
│   ├── A_input_sentence_hist.png
│   ├── B_bullet_hist.png
│   ├── C_codes_per_sample_hist.png
│   └── ...
└── README.md

```

---

## Results

- **Leaderboard Best Score**: `39.57 / 60`  
  - Strong quantitative stability  
  - Improved LLM-based Conciseness & Clinical Clarity  
  - ICD-10 accuracy remains the main improvement area  

---
