# Automated Discharge Summary Generation

Deterministic Rule-Based Clinical Text Structuring Pipeline

# Competition Information

M.A.R.S. 2025 Discharge Summary Generation Datathon
(Medical Auto-documentation with Real-world Structuring)

Organizer:
Medical AI Center · Big Data Center,
Seoul National University Bundang Hospital (SNUBH)

Competition Period:
September 16, 2025 – October 17, 2025

Task:
Automated generation of structured discharge summaries from real-world EMR-style data,
following a fixed guideline and strict data usage constraints.

Result:
6th Place in the Final Round

This project reflects the full workflow and methodology used throughout both the preliminary and final rounds of the competition.
The public repository is provided in a compliance-aware, non-identifiable form, independent of ranking outcomes.

## Overview

This repository documents a deterministic, rule-based system for automatically generating discharge summaries from inpatient clinical records.
The system operates strictly within the admission-to-discharge time window and produces summaries that follow a predefined, fixed template.

This project was developed in the context of a medical AI datathon and is presented here as a **public portfolio version**.
No real patient data, clinical text, diagnoses, laboratory values, dates, or hospital-specific information are included in this repository.
All examples are **synthetic and reconstructed for illustrative purposes only**.

---

## Task Definition

* Input: Multiple EMR-style tables
  (admission, diagnosis, surgery/anesthesia, nursing notes, medical notes, chief complaint)
* Output: One discharge summary per case, generated automatically by code
* Execution: Final code executed by the organizers in a closed environment
* Evaluation dimensions:

  * Guideline compliance
  * Factual consistency
  * Internal coherence
  * Clinical usefulness
  * Clarity and conciseness
  * Runtime efficiency

---

## Output Format

Each generated discharge summary strictly contains the following five slots, in the fixed order:

1. Reason for Admission and Medical History
2. Hospital Course
3. Discharge Outcome
4. Key Test Results
5. Patient Summary

If a slot cannot be populated, a predefined placeholder text is automatically inserted to preserve format compliance.

---

## System Architecture

### Three-Stage Deterministic Pipeline

### Stage A: Fact Extraction and Normalization

* Unified time filtering across all tables
  (admission date ≤ record time ≤ discharge date)
* Slot-specific source mapping:

  * Chief complaint → admission reason
  * Diagnosis table → diagnosis summary
  * Surgery/anesthesia table → procedure information
  * Nursing and medical notes → hospital course and test cues
* Keyword- and regular-expression–based extraction
* Test results categorized into:

  * Laboratory
  * Imaging
  * Functional
  * Pathology
* Duplicate test items reduced to the most recent one or two entries

---

### Stage B: Text Construction (Optional)

* Default configuration: LLM disabled
* Sentence generation performed using deterministic rules
* Optional hooks preserved for future local LLM integration

This ensures offline execution, no randomness, and full reproducibility.

---

### Stage C: Guardrails and Export

* Fixed five-slot template enforcement
* Automatic insertion of missing-slot placeholders
* Explicit exclusion of future-plan expressions
* Global length constraints for readability
* Automatic export of:

  * result.csv
  * runtime.txt

---

## Label Normalization and MeSH Integration

To reduce ambiguity and redundancy caused by synonym usage:

* Multiple aliases are mapped to canonical labels
* Public examples use placeholder labels only
* If MeSH resources are available, synonym coverage is expanded
* If not, the system automatically falls back to an internal dictionary

The execution logic remains identical regardless of resource availability.

---

## Compliance and Ethics

* No real patient data is included
* No actual diagnoses, laboratory names, values, or dates are disclosed
* All figures and tables are synthetic
* Patient-level references are replaced with “case examples”
* Hospital names and internal schemas are omitted

This repository is intended solely to document **system design and methodological approach**.

---

## Reproducibility

* No stochastic components
* Identical input produces identical output
* No external API usage
* Fully executable in an offline environment

Key functions:

* run_submit
* run_submit_batch
* format_with_template_5
* within_stay
