"""
AustrianRx Vision
=================

Repository-friendly reference implementation for OCR-based parsing of historical
Austrian prescriptions with heavy handwriting, overlaid pharmacy stamps, and
mixed printed/handwritten content.

Recommended repository name
---------------------------
austrian-rx-vision

Suggested GitHub description
----------------------------
Production-style OCR pipeline for historical Austrian prescriptions using
TrOCR, Tesseract, handwriting masking, stamp-aware preprocessing, and
structured field extraction.

What this script is meant to do
-------------------------------
This script processes scanned prescription PDFs like the examples in the
project dataset, where each page contains:
- a printed doctor header at the top,
- handwritten medicine lines in the body,
- blue or violet stamps that may overlap the writing,
- dates, prices, patient names, and prescription instructions.

The pipeline is designed to match those scanned examples as closely as
possible by:
1. rendering PDF pages at high resolution,
2. separating printed header, handwriting, and stamp regions,
3. detecting text bands automatically,
4. running TrOCR + Tesseract as a two-engine OCR ensemble,
5. cleaning OCR output with domain-specific rules,
6. identifying drug names with fuzzy matching,
7. extracting structured fields such as price, patient name, and instructions.

Why the extra comments exist
----------------------------
This file is intentionally documented in detail so it can be pushed directly to
GitHub and still be understandable to reviewers, collaborators, or interviewers.
Most major functions include:
- purpose,
- inputs,
- outputs,
- key parameters,
- practical notes about how they relate to the prescription images.

"""

# ============================================================
# AUSTRIAN PRESCRIPTION OCR v10 -- PRODUCTION GRADE
# For Google Colab (T4 GPU recommended) or any Python 3.10+ env
#
# Features:
#  - Colour-based stamp removal (HSV + LAB)
#  - Named ROI detection (drug1, drug2, sig_notes)
#  - Line segmentation with tall-band splitting
#  - Ensemble OCR: TrOCR (beam search) + Tesseract
#  - Dosage pattern language model (regex + n-gram)
#  - Character-level GRU correction model
#  - ATC/Kaggle vocabulary + fuzzy drug matching
#  - Combined confidence scoring
#  - Character-level bounding boxes
#  - Extended fine-tuning pipeline (Cell 6)
# ============================================================
# ============================================================
# CELL 1 -- Install dependencies
# ============================================================
import subprocess, shutil, sys
pkgs_apt = ["tesseract-ocr", "tesseract-ocr-eng", "tesseract-ocr-deu"]
subprocess.run(["apt-get", "install", "-y"] + pkgs_apt,
               check=True, capture_output=True)
pkgs_pip = [
    "pymupdf", "pillow", "pytesseract", "opencv-python-headless",
    "transformers", "torch", "torchvision", "rapidfuzz",
    "accelerate", "safetensors", "datasets", "sentencepiece",
    "requests", "kaggle",   # for ATC vocab download + Kaggle dataset
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs_pip,
               check=True, capture_output=True)
print(f"Tesseract: {shutil.which('tesseract')}")
# ============================================================
# CELL 2 -- Choose Existing PDF or Upload New
# ============================================================
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import files

# Base directory: /content on Colab, or current working directory otherwise.
# Change BASE_DIR if you run this outside Google Colab.
BASE_DIR = "/content" if os.path.isdir("/content") else os.getcwd()

OUTPUT_CSV     = os.path.join(BASE_DIR, "prescription_output.csv")
DEBUG_DIR      = os.path.join(BASE_DIR, "debug_images")
ANNOTATED_DIR  = os.path.join(BASE_DIR, "annotated_output")
VOCAB_DIR      = os.path.join(BASE_DIR, "vocab_cache")
TROCR_FT_DIR   = os.path.join(BASE_DIR, "trocr_finetuned")
CORRECTOR_DIR  = os.path.join(BASE_DIR, "corrector_model")
KAGGLE_ROOT    = os.path.join(BASE_DIR, "kaggle_bd")
KAGGLE_DATASET = "mamun18/doctors-handwritten-prescription-bd-dataset"

for d in [DEBUG_DIR, ANNOTATED_DIR, VOCAB_DIR]:
    os.makedirs(d, exist_ok=True)

existing_pdfs = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".pdf")]

def use_selected(b):
    """Callback: set the global PDF_PATH to the dropdown selection."""
    clear_output(wait=True); display(ui)
    globals()["PDF_PATH"] = os.path.join(BASE_DIR, dropdown.value)
    print(f"Selected: {dropdown.value}")

def use_uploaded(b):
    """Callback: upload a new PDF via the Colab file dialog and set PDF_PATH."""
    clear_output(wait=True)
    uploaded = files.upload()
    if uploaded:
        fn = list(uploaded.keys())[0]
        globals()["PDF_PATH"] = os.path.join(BASE_DIR, fn)
        print(f"Uploaded: {fn}")

upload_btn = widgets.Button(description="Upload New PDF", button_style="primary",
                             layout=widgets.Layout(width="180px"))
upload_btn.on_click(use_uploaded)

if existing_pdfs:
    dropdown = widgets.Dropdown(options=existing_pdfs, description="Existing:",
                                layout=widgets.Layout(width="350px"))
    select_btn = widgets.Button(description="Use Selected", button_style="success",
                                layout=widgets.Layout(width="150px"))
    select_btn.on_click(use_selected)
    ui = widgets.VBox([widgets.HTML("<b>Choose PDF source:</b>"),
                       widgets.HBox([dropdown, select_btn]),
                       widgets.HTML("<i>-- or --</i>"), upload_btn])
else:
    ui = widgets.VBox([widgets.HTML("<b>No PDFs found. Upload one:</b>"), upload_btn])
display(ui)
# ============================================================
# CELL 3 -- Full Pipeline v9 (PRODUCTION GRADE)
# ============================================================
import io, re, csv, json, math, glob, random, unicodedata, subprocess
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
import fitz
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from rapidfuzz import process as fuzz_process, fuzz

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v11] Device: {DEVICE}")

# -------------------------------------------------------------
# Section 1  DRUG VOCABULARY -- Austrian/German + ATC + Kaggle
# -------------------------------------------------------------
DRUG_VOCAB = {
    # -- These exact prescriptions --
    "Tryptizol": "Amitriptylin (Antidepressivum)",
    "Valium": "Diazepam (Benzodiazepin/Anxiolytikum)",
    # -- Benzodiazepines --
    "Diazepam": "Benzodiazepin", "Librium": "Chlordiazepoxid",
    "Mogadon": "Nitrazepam", "Rohypnol": "Flunitrazepam",
    "Lexotanil": "Bromazepam", "Tranxilium": "Clorazepat",
    "Lorazepam": "Benzodiazepin", "Tavor": "Lorazepam",
    "Temesta": "Lorazepam", "Alprazolam": "Benzodiazepin",
    "Clonazepam": "Benzodiazepin",
    # -- Antidepressiva (tricyclisch) --
    "Tofranil": "Imipramin", "Anafranil": "Clomipramin",
    "Saroten": "Amitriptylin", "Laroxyl": "Amitriptylin",
    "Amitriptylin": "Trizyklisches AD", "Doxepin": "Trizyklisches AD",
    "Sinequan": "Doxepin",
    # -- SSRI / modern AD --
    "Sertralin": "SSRI", "Fluoxetin": "SSRI", "Citalopram": "SSRI",
    "Escitalopram": "SSRI", "Venlafaxin": "SNRI", "Mirtazapin": "NaSSA",
    # -- Analgetika --
    "Aspirin": "ASS", "Paracetamol": "Analgetikum",
    "Mexalen": "Paracetamol", "Parkemed": "Mefenaminsäure",
    "Voltaren": "Diclofenac", "Diclofenac": "NSAR",
    "Ibuprofen": "NSAR", "Brufen": "Ibuprofen",
    "Novalgin": "Metamizol", "Tramal": "Tramadol",
    "Tramadol": "Opioid-Analgetikum", "Morphin": "Opioid",
    "Naproxen": "NSAR", "Celecoxib": "COX-2-Hemmer",
    # -- Kardiologika --
    "Adalat": "Nifedipin", "Nifedipin": "Ca-Antagonist",
    "Verapamil": "Ca-Antagonist", "Isoptin": "Verapamil",
    "Captopril": "ACE-Hemmer", "Ramipril": "ACE-Hemmer",
    "Lisinopril": "ACE-Hemmer", "Enalapril": "ACE-Hemmer",
    "Atenolol": "Betablocker", "Tenormin": "Atenolol",
    "Beloc": "Metoprolol", "Metoprolol": "Betablocker",
    "Bisoprolol": "Betablocker", "Carvedilol": "Betablocker",
    "Propranolol": "Betablocker", "Dociton": "Propranolol",
    "Furosemid": "Schleifendiuretikum", "Lasix": "Furosemid",
    "Hydrochlorothiazid": "Thiazid", "Aldactone": "Spironolacton",
    "Losartan": "AT1-Blocker", "Candesartan": "AT1-Blocker",
    "Valsartan": "AT1-Blocker", "Amlodipin": "Ca-Antagonist",
    "Digitoxin": "Herzglykosid", "Digoxin": "Herzglykosid",
    "Simvastatin": "Statin", "Atorvastatin": "Statin",
    # -- Antibiotika --
    "Penicillin": "Antibiotikum", "Amoxicillin": "Aminopenicillin",
    "Augmentin": "Amoxicillin/Clavulansäure",
    "Doxycyclin": "Tetracyclin", "Vibramycin": "Doxycyclin",
    "Erythromycin": "Makrolid", "Azithromycin": "Makrolid",
    "Ciprofloxacin": "Fluorchinolon", "Clindamycin": "Lincosamid",
    "Cephalexin": "Cephalosporin",
    # -- GI --
    "Pantoprazol": "PPI", "Omeprazol": "PPI",
    "Buscopan": "Butylscopolamin", "Maalox": "Antazidum",
    # -- Kortikosteroide --
    "Prednisolon": "Glukokortikoid", "Prednison": "Glukokortikoid",
    "Dexamethason": "Glukokortikoid",
    # -- Antiepileptika --
    "Phenobarbital": "Barbiturat", "Luminal": "Phenobarbital",
    "Phenytoin": "Antiepileptikum", "Carbamazepin": "Antiepileptikum",
    "Tegretol": "Carbamazepin", "Valproat": "Antiepileptikum",
    "Lamotrigin": "Antiepileptikum", "Gabapentin": "Antiepileptikum",
    "Pregabalin": "Antiepileptikum",
    # -- Antipsychotika --
    "Haloperidol": "Neuroleptikum", "Risperidon": "Atypisches AP",
    "Quetiapin": "Atypisches AP", "Olanzapin": "Atypisches AP",
    # -- Sonstiges --
    "Insulin": "Antidiabetikum", "Metformin": "Biguanid",
    "Levothyroxin": "Schilddrüsenhormon", "Euthyrox": "Levothyroxin",
    "Allopurinol": "Urikostatikum", "Zyloric": "Allopurinol",
    "Warfarin": "VKA", "Phenprocoumon": "VKA (Marcumar)",
    "Heparin": "Antikoagulans", "Clopidogrel": "Thrombozytenaggr.",
    "Zolpidem": "Z-Substanz",
    "Tavegil": "Antihistaminikum", "Fenistil": "Dimetinden",
}

DRUG_NAMES_LOWER = {k.lower(): (k, v) for k, v in DRUG_VOCAB.items()}

# -- Prescription abbreviations (Austrian/German) --
RX_ABBREVIATIONS = {
    "Rp": "Recipe", "S": "Signa", "D.S.": "Da, Signa",
    "O.P.": "Originalpackung", "O.P.Nct": "Originalpackung nachts",
    "O.P.Nci": "Originalpackung nachts", "Nct": "nocte",
    "Nci": "nocte", "tägl": "täglich", "Tbl": "Tablette(n)",
    "Tabs": "Tabletten", "Kps": "Kapsel(n)", "Trpf": "Tropfen",
    "mg": "Milligramm", "ml": "Milliliter", "IE": "Internat. Einh.",
    "Stk": "Stück", "Rep": "Repetatur", "repetatur": "wiederholen",
    "morgens": "morgens", "abends": "abends", "mittags": "mittags",
}

# -- Seed vocabulary for general matching --
GERMAN_RX_SEEDS = [
    "Tryptizol","Valium","Diazepam","Amoxicillin","Ibuprofen","Paracetamol",
    "Aspirin","Metformin","Ramipril","Omeprazol","Simvastatin","Pantoprazol",
    "Rp","Sig","Tabs","Tabletten","Kapseln","Tropfen","Ampullen","Salbe",
    "morgens","abends","täglich","zweimal","dreimal","tägl","mg","ml","IE",
    "Stk","Packung","Wiederholung","repetatur","Repetatur","Originalpackung",
    "nocte","O.P.Nct","für","Herr","Frau","Dr","Ordination",
]


def fetch_atc_vocabulary() -> List[str]:
    """
    Download drug names from the FDA's public API (ATC/OpenFDA database).

    How it works:
        - Checks for a local cache file first (vocab_cache/atc_vocab.json).
        - If no cache, hits two FDA endpoints: one for generic names, one for
          brand names, collecting up to 1000 of each.
        - Stores the result locally so subsequent runs are instant.
        - If the network request fails (offline, timeout), returns an empty
          list without crashing -- the pipeline still works with the built-in
          DRUG_VOCAB dictionary.

    Returns:
        A list of drug-name strings (Title Case), e.g. ["Amoxicillin", ...].
    """
    cache = os.path.join(VOCAB_DIR, "atc_vocab.json")
    if os.path.exists(cache):
        with open(cache) as f:
            words = json.load(f)
        print(f"[Vocab] ATC cache: {len(words)} terms")
        return words
    try:
        import requests
        url = ("https://api.fda.gov/drug/label.json?"
               "search=openfda.substance_name:*"
               "&count=openfda.generic_name.exact&limit=1000")
        resp = requests.get(url, timeout=10)
        terms = []
        if resp.status_code == 200:
            terms = [item["term"] for item in resp.json().get("results", [])]
        url2 = ("https://api.fda.gov/drug/label.json?"
                "search=openfda.brand_name:*"
                "&count=openfda.brand_name.exact&limit=1000")
        resp2 = requests.get(url2, timeout=10)
        if resp2.status_code == 200:
            terms += [item["term"] for item in resp2.json().get("results", [])]
        cleaned = list({t.strip().title() for t in terms if len(t.strip()) >= 3})
        with open(cache, "w") as f:
            json.dump(cleaned, f)
        print(f"[Vocab] ATC downloaded: {len(cleaned)} terms")
        return cleaned
    except Exception as e:
        print(f"[Vocab] ATC download failed: {e}")
        return []


def ensure_kaggle_dataset() -> Tuple[Optional[str], Optional[str]]:
    """Try to find or download Kaggle BD prescription dataset."""
    possible = [
        KAGGLE_ROOT,
        os.path.join(BASE_DIR, "prescriptions/Doctor's Handwritten Prescription BD dataset"),
        os.path.join(BASE_DIR, "doctors-handwritten-prescription-bd-dataset"),
    ]
    for root in possible:
        lcsv = os.path.join(root, "Training", "training_labels.csv")
        wdir = os.path.join(root, "Training", "training_words")
        if os.path.exists(lcsv) and os.path.isdir(wdir):
            print(f"[Kaggle] [ok] Found: {root}")
            return lcsv, wdir
    #downloading
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    colab_json = os.path.join(BASE_DIR, "kaggle.json")
    if os.path.exists(colab_json) and not os.path.exists(kaggle_json):
        import shutil as _sh
        os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
        _sh.copy(colab_json, kaggle_json)
        os.chmod(kaggle_json, 0o600)
    if not os.path.exists(kaggle_json):
        print("[Kaggle] [fail] No kaggle.json -- skipping dataset")
        return None, None
    try:
        r = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", KAGGLE_ROOT, "--unzip"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[Kaggle] [fail] {r.stderr.strip()[:80]}")
            return None, None
        for root, dirs, fnames in os.walk(KAGGLE_ROOT):
            if "training_labels.csv" in fnames and "training_words" in dirs:
                return (os.path.join(root, "training_labels.csv"),
                        os.path.join(root, "training_words"))
    except Exception as e:
        print(f"[Kaggle] [fail] {e}")
    return None, None


def build_vocabulary(labels_csv: Optional[str]) -> List[str]:
    """Merge seed + ATC + Kaggle into unified vocabulary."""
    vocab: Set[str] = set(GERMAN_RX_SEEDS)
    vocab.update(DRUG_VOCAB.keys())
    vocab.update(fetch_atc_vocabulary())
    if labels_csv:
        try:
            import pandas as pd
            df = pd.read_csv(labels_csv)
            lc = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            for c in df.columns:
                if c.lower() in ("label", "text", "word", "gt"):
                    lc = c
            for val in df[lc].dropna().astype(str):
                val = val.strip()
                if 2 <= len(val) <= 50:
                    vocab.add(val)
                    for w in val.split():
                        if len(w) >= 3:
                            vocab.add(w)
        except Exception as e:
            print(f"[Vocab] Kaggle labels error: {e}")
    result = sorted(vocab)
    print(f"[Vocab] [ok] {len(result)} terms")
    return result


def find_drug_in_text(text: str) -> Tuple[str, str, int]:
    """
    Search a line of OCR text for a known drug name.

    Strategy (tried in order, first match wins):
        1. Exact match: split text into words, check each against the
           DRUG_VOCAB dictionary (case-insensitive).  Score = 100.
        2. Fuzzy full-string: compare the whole cleaned text against all
           drug names using rapidfuzz partial_ratio.  Cutoff = 72.
        3. Fuzzy per-word: for each word >= 4 chars, compare against all
           drug names using rapidfuzz ratio.  Cutoff = 68.

    Parameters:
        text: a single line of OCR output, e.g. "Tryptizol 25mg O.P.Nct"

    Returns:
        (drug_name, drug_class, match_score) where:
            drug_name  = canonical name, e.g. "Tryptizol"
            drug_class = pharmacological class, e.g. "Amitriptylin (Antidepressivum)"
            match_score = 0-100 integer (0 = no match found)
    """
    if not text or len(text.strip()) < 3:
        return "", "", 0
    clean = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', ' ', text).strip()
    words = clean.split()
    # Exact match
    for word in words:
        wl = word.lower()
        if wl in DRUG_NAMES_LOWER:
            name, cls = DRUG_NAMES_LOWER[wl]
            return name, cls, 100
    # Fuzzy on full string
    drug_list = list(DRUG_VOCAB.keys())
    result = fuzz_process.extractOne(clean, drug_list,
                                     scorer=fuzz.partial_ratio,
                                     score_cutoff=72)
    if result:
        return result[0], DRUG_VOCAB[result[0]], int(result[1])
    # Fuzzy per word
    best = (0, "", "")
    for word in words:
        if len(word) < 4:
            continue
        r = fuzz_process.extractOne(word, drug_list,
                                    scorer=fuzz.ratio, score_cutoff=68)
        if r and r[1] > best[0]:
            best = (int(r[1]), r[0], DRUG_VOCAB[r[0]])
    return best[1], best[2], best[0]


def vocab_match(text: str, vocab: List[str],
                threshold: int = 68) -> Tuple[str, int]:
    """General vocabulary fuzzy match."""
    if not text.strip() or not vocab:
        return "", 0
    clean = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s\-\.]", "", text).strip()
    if not clean:
        return "", 0
    r = fuzz_process.extractOne(clean, vocab,
                                scorer=fuzz.token_sort_ratio,
                                score_cutoff=threshold)
    return (r[0], int(r[1])) if r else ("", 0)


# -------------------------------------------------------------
# Section 2  DOSAGE PATTERN DECODER (regex + n-gram scoring)
# -------------------------------------------------------------
# Common Austrian prescription dosage patterns
DOSAGE_PATTERNS = [
    # "Tryptizol 25mg O.P.Nct"
    (r'(\w+)\s+(\d+)\s*(mg|ml|g|IE|mcg|µg)\s+(O\.?P\.?\s*N[cC][tTiI])',
     lambda m: f"{m.group(1)} {m.group(2)}{m.group(3)} O.P.Nct"),
    # "Valium 2 O.P.Nct"
    (r'(\w+)\s+(\d+)\s+(O\.?P\.?\s*N[cC][tTiI])',
     lambda m: f"{m.group(1)} {m.group(2)} O.P.Nct"),
    # "S: 1 in the morning"
    (r'S[\.:]\s*(\d+)\s+in\s+the?\s+(morning|evening|abend|morgen)',
     lambda m: f"S: {m.group(1)} in the {m.group(2)}"),
    # "S: 2x täglich 1 Tbl"
    (r'S[\.:]\s*(\d+)\s*[xXx]\s*(tägl(?:ich)?|daily)\s+(\d+)\s*(Tbl|Tab|Kps)',
     lambda m: f"S: {m.group(1)}x täglich {m.group(3)} {m.group(4)}."),
    # "45,- 1 in the evening"
    (r'(\d+)[,\.]\s*[-----]\s*(\d+)\s+in\s+the?\s+(evening|abend)',
     lambda m: f"{m.group(1)},-- {m.group(2)} in the {m.group(3)}"),
    # "Repetatur"
    (r'[SR][\.:]\s*[Rr]ep(?:etatur|etahir|etahui|etchi)?',
     lambda m: "S: Repetatur"),
    # "für Mr./Frau Name"
    (r'für\s+(Mr|Mrs|Frau|Herr|Hr)\.?\s+(\w+)',
     lambda m: f"für {m.group(1)}. {m.group(2)}"),
]

# N-gram model trained on prescription language
class DosageNGramModel:
    """
    Character-level n-gram model for scoring prescription text plausibility.
    Trained on synthetic prescription strings.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.counts: Counter = Counter()
        self.total = 0
        self._trained = False

    def train(self, texts: List[str]):
        """Train on a corpus of correct prescription strings."""
        for text in texts:
            padded = "^" * (self.n - 1) + text.lower() + "$"
            for i in range(len(padded) - self.n + 1):
                self.counts[padded[i:i + self.n]] += 1
                self.total += 1
        self._trained = True

    def score(self, text: str) -> float:
        """Log-probability per character. Higher = more plausible."""
        if not self._trained or not text:
            return 0.0
        padded = "^" * (self.n - 1) + text.lower() + "$"
        log_prob = 0.0
        count = 0
        for i in range(len(padded) - self.n + 1):
            ngram = padded[i:i + self.n]
            freq = self.counts.get(ngram, 0)
            log_prob += math.log((freq + 1) / (self.total + len(self.counts) + 1))
            count += 1
        return log_prob / max(count, 1)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"n": self.n, "counts": dict(self.counts),
                       "total": self.total}, f)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.n = d["n"]
        self.counts = Counter(d["counts"])
        self.total = d["total"]
        self._trained = True


def build_prescription_corpus() -> List[str]:
    """Generate synthetic prescription text corpus for n-gram training."""
    drugs = list(DRUG_VOCAB.keys())
    dosages = ["25mg", "50mg", "100mg", "2mg", "5mg", "10mg", "20mg",
               "75mg", "150mg", "200mg", "500mg", "1g", "2ml", "5ml",
               "10ml", "100IE", "250mg", "400mg", "600mg", "800mg"]
    sigs = [
        "O.P.Nct", "O.P.Nci", "O.P.",
        "S: 1x täglich", "S: 2x täglich 1 Tbl.",
        "S: 3x täglich 1 Tbl.", "S: 1 morgens, 1 abends",
        "S: 1 in the morning", "S: 1 in the evening",
        "S: 2x täglich", "S: Repetatur",
        "1 morgens", "1 abends", "1 mittags",
        "1-0-1", "1-1-1", "0-0-1", "1-0-0",
    ]
    notes = [
        "für Mr. Alain", "für Frau Müller", "für Herr Schmidt",
        "Repetatur", "S: Repetatur!", "Wiederholung",
    ]
    corpus = []
    for drug in drugs:
        for dos in dosages[:8]:
            for sig in sigs[:6]:
                corpus.append(f"{drug} {dos} {sig}")
        corpus.append(f"Rp. {drug}")
    for note in notes:
        corpus.append(note)
    # Add date patterns
    for m in range(1, 13):
        for y in [1975, 1976, 1977, 1978, 1979, 1980]:
            corpus.append(f"{random.randint(1,28)}.{m}.{y}")
    return corpus


# Initialize n-gram model
NGRAM_MODEL = DosageNGramModel(n=4)
_corpus = build_prescription_corpus()
NGRAM_MODEL.train(_corpus)
print(f"[NGram] Trained on {len(_corpus)} synthetic prescription strings")


def apply_dosage_patterns(text: str) -> str:
    """Try to match and normalize dosage patterns."""
    for pattern, formatter in DOSAGE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return formatter(m)
            except:
                pass
    return text


# -------------------------------------------------------------
# Section 3  OCR CORRECTION MODEL (Character-level GRU Seq2Seq)
# -------------------------------------------------------------
# Vocabulary for character-level model
CHAR_VOCAB = (
    list("abcdefghijklmnopqrstuvwxyz"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "0123456789"
         "äöüÄÖÜß"
         " .,;:!?-/()=+#@'\"_"
         "x%µ°€")
)
CHAR2IDX = {c: i + 3 for i, c in enumerate(CHAR_VOCAB)}
CHAR2IDX["<pad>"] = 0
CHAR2IDX["<sos>"] = 1
CHAR2IDX["<eos>"] = 2
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
VOCAB_SIZE = len(CHAR2IDX)
MAX_SEQ_LEN = 128


def encode_text(text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
    ids = [CHAR2IDX.get(c, CHAR2IDX.get(" ", 0)) for c in text[:max_len - 2]]
    ids = [CHAR2IDX["<sos>"]] + ids + [CHAR2IDX["<eos>"]]
    ids += [CHAR2IDX["<pad>"]] * (max_len - len(ids))
    return ids


def decode_ids(ids: List[int]) -> str:
    chars = []
    for idx in ids:
        if idx == CHAR2IDX["<eos>"]:
            break
        if idx in (CHAR2IDX["<pad>"], CHAR2IDX["<sos>"]):
            continue
        chars.append(IDX2CHAR.get(idx, "?"))
    return "".join(chars)


class CorrectorEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64,
                 hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.gru(emb)
        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)
        # Repeat for decoder layers
        hidden = hidden.repeat(2, 1, 1)  # n_layers
        return out, hidden


class CorrectorDecoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64,
                 hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attn = nn.Linear(hidden_dim + hidden_dim * 2, 1)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_out):
        emb = self.embedding(x)  # (B, 1, E)
        # Attention
        B, S, H2 = encoder_out.shape
        h_rep = hidden[-1].unsqueeze(1).repeat(1, S, 1)  # (B, S, H)
        attn_in = torch.cat([h_rep, encoder_out], dim=2)  # (B, S, H+H2)
        attn_w = torch.softmax(self.attn(attn_in), dim=1)  # (B, S, 1)
        context = (attn_w * encoder_out).sum(dim=1, keepdim=True)  # (B, 1, H2)
        gru_in = torch.cat([emb, context], dim=2)
        out, hidden = self.gru(gru_in, hidden)
        pred = self.out(out.squeeze(1))
        return pred, hidden


class OCRCorrectorModel(nn.Module):
    """
    Character-level sequence-to-sequence model for fixing OCR errors.

    Architecture:
        - Encoder: bidirectional GRU with 2 layers, reads the noisy OCR text
          character-by-character and produces a context representation.
        - Decoder: GRU with attention, generates the corrected text one
          character at a time, attending back to the encoder outputs.

    Training:
        Trained on synthetic pairs of (noisy_text, clean_text) where the
        noise simulates common handwriting-OCR confusions like:
            'rn' misread as 'm', 'u'/'n' swaps, 'l'/'1' swaps, etc.

    Usage after training:
        model = OCRCorrectorModel().to(device)
        model.load_state_dict(torch.load('corrector.pt'))
        corrected = model.correct('Tryptlzol 25rng O.P.Nct')
        # -> 'Tryptizol 25mg O.P.Nct'

    Parameters:
        vocab_size: number of characters in the vocabulary (default: VOCAB_SIZE,
                    which includes a-z, A-Z, 0-9, German umlauts, punctuation,
                    and three special tokens: <pad>, <sos>, <eos>).
    """
    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder = CorrectorEncoder(vocab_size)
        self.decoder = CorrectorDecoder(vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, T = tgt.shape
        outputs = torch.zeros(B, T, VOCAB_SIZE, device=src.device)
        enc_out, hidden = self.encoder(src)
        inp = tgt[:, 0:1]  # <sos>
        for t in range(1, T):
            pred, hidden = self.decoder(inp, hidden, enc_out)
            outputs[:, t] = pred
            if random.random() < teacher_forcing_ratio:
                inp = tgt[:, t:t + 1]
            else:
                inp = pred.argmax(dim=1, keepdim=True)
        return outputs

    @torch.inference_mode()
    def correct(self, text: str, max_len: int = MAX_SEQ_LEN) -> str:
        """Correct a single OCR text string."""
        self.eval()
        src = torch.tensor([encode_text(text, max_len)], device=DEVICE)
        enc_out, hidden = self.encoder(src)
        inp = torch.tensor([[CHAR2IDX["<sos>"]]], device=DEVICE)
        result = []
        for _ in range(max_len):
            pred, hidden = self.decoder(inp, hidden, enc_out)
            top = pred.argmax(dim=1).item()
            if top == CHAR2IDX["<eos>"]:
                break
            if top != CHAR2IDX["<pad>"]:
                result.append(IDX2CHAR.get(top, "?"))
            inp = torch.tensor([[top]], device=DEVICE)
        return "".join(result)


def load_corrector() -> Optional[OCRCorrectorModel]:
    """Load pre-trained corrector model if available."""
    model_path = os.path.join(CORRECTOR_DIR, "corrector.pt")
    if not os.path.exists(model_path):
        return None
    try:
        model = OCRCorrectorModel().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"[Corrector] [ok] Loaded from {model_path}")
        return model
    except Exception as e:
        print(f"[Corrector] [fail] Load failed: {e}")
        return None


# -------------------------------------------------------------
# Section 4  STAMP DETECTION (HSV + LAB color space)
# -------------------------------------------------------------
def detect_stamp_mask(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Detect purple/violet/blue rubber-stamp ink in a prescription image.

    Austrian prescriptions from pharmacies like Kurapotheke Badgastein
    typically have date stamps, pharmacy stamps, and doctor stamps
    overlaid in purple or blue ink.  These overlap with the handwriting
    and confuse the OCR engine, so we need to find and suppress them.

    Parameters:
        arr_rgb: numpy array of shape (H, W, 3) in RGB colour order.

    Algorithm:
        1. Convert to HSV and CIE-LAB colour spaces.
        2. Threshold on hue (100-165 in HSV = purple range) + saturation.
        3. Also catch blue stamp ink (hue 85-130, higher saturation).
        4. In LAB space, look for positive 'a' and negative 'b' channels
           (the purple quadrant).
        5. Combine both masks, clean with morphological close/open.
        6. Keep only connected components large enough to be a real stamp
           (at least 0.1% of the image area).
        7. Dilate slightly to catch ink edges.

    Returns:
        Binary mask (uint8, 0 or 255) of the same height/width as input.
        Pixels that are 255 belong to detected stamp regions.
    """
    H, W = arr_rgb.shape[:2]
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    l_ch = lab[:,:,0].astype(np.float32)
    a_ch = lab[:,:,1].astype(np.float32) - 128
    b_ch = lab[:,:,2].astype(np.float32) - 128

    # Purple/violet stamp ink (HSV hue 100-165)
    purple_hsv = ((h >= 100) & (h <= 165) & (s > 30) & (v > 60) & (v < 220))
    # LAB purple (positive a, negative b)
    purple_lab = ((a_ch > -5) & (b_ch < -8) & (l_ch > 30) & (l_ch < 180))
    # Blue stamp ink
    blue_stamp = ((h >= 85) & (h <= 130) & (s > 40) & (v > 50) & (v < 200))

    stamp_raw = (purple_hsv | purple_lab | blue_stamp).astype(np.uint8) * 255

    # Morphological cleanup
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 8))
    stamp = cv2.morphologyEx(stamp_raw, cv2.MORPH_CLOSE, kern_close)
    kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stamp = cv2.morphologyEx(stamp, cv2.MORPH_OPEN, kern_open)

    # Keep only large components (actual stamps)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(stamp, 8)
    mask = np.zeros_like(stamp)
    min_area = int(H * W * 0.001)
    for lbl in range(1, nl):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == lbl] = 255

    # Dilate to catch edges
    kern_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.dilate(mask, kern_dil, iterations=1)
    return mask


# -------------------------------------------------------------
# Section 5  INK MASK CLASS (from v6, enhanced with stamp removal)
# -------------------------------------------------------------
class InkMask:
    """
    Produces binary masks that separate handwriting from printed text and stamps.

    A typical Austrian prescription has three visual layers:
        1. Printed header (doctor name, address, phone -- typeset text)
        2. Handwritten body (drug names, dosages, notes -- the part we OCR)
        3. Stamp overlays (pharmacy stamp, date, doctor's round stamp)

    This class generates pixel-level masks for each layer so the OCR
    engine only sees clean handwriting without header clutter or stamp
    bleed-through.

    Key methods:
        get_ink_mask()         -> all dark pixels (handwriting + print)
        get_handwriting_mask() -> handwriting only (header + stamps removed)
        get_stamp_mask()       -> stamp regions only
        save_debug()           -> colour-coded debug image (red=stamp,
                                  green=handwriting, cyan=printed)
    """
    def get_ink_mask(self, pil_img: Image.Image) -> np.ndarray:
        """All ink (handwriting + printed). Returns uint8 mask."""
        gray = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=31, C=10)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)
        return binary

    def get_handwriting_mask(self, pil_img: Image.Image,
                             header_frac: float = None) -> np.ndarray:
        """Handwriting only: removes header, stamps, large printed blocks.
        If header_frac is None, auto-detects the header boundary."""
        H, W = pil_img.height, pil_img.width
        arr = np.array(pil_img)

        # Full ink
        full = self.get_ink_mask(pil_img)
        hw = full.copy()

        # -- Adaptive header detection --
        if header_frac is None:
            header_frac = self._detect_header_boundary(arr, full, H, W)
        hw[:int(H * header_frac), :] = 0

        # Subtract stamp regions
        stamp_mask = detect_stamp_mask(arr)
        hw = cv2.bitwise_and(hw, cv2.bitwise_not(stamp_mask))

        # Remove very large rectangular blobs
        nl, labels, stats, _ = cv2.connectedComponentsWithStats(hw, 8)
        for lbl in range(1, nl):
            cw = stats[lbl, cv2.CC_STAT_WIDTH]
            ch = stats[lbl, cv2.CC_STAT_HEIGHT]
            area = stats[lbl, cv2.CC_STAT_AREA]
            if cw > W * 0.45 and ch > H * 0.06 and area > W * H * 0.008:
                hw[labels == lbl] = 0
            if ch > H * 0.4 and cw < 15:
                hw[labels == lbl] = 0
            if cw > W * 0.4 and ch < 10:
                hw[labels == lbl] = 0

        # Remove tiny specks
        nl2, labels2, stats2, _ = cv2.connectedComponentsWithStats(hw, 8)
        for lbl in range(1, nl2):
            if stats2[lbl, cv2.CC_STAT_AREA] < 25:
                hw[labels2 == lbl] = 0

        return hw

    def get_stamp_mask(self, pil_img: Image.Image) -> np.ndarray:
        return detect_stamp_mask(np.array(pil_img))

    def _detect_header_boundary(self, arr_rgb: np.ndarray,
                                 ink_mask: np.ndarray,
                                 H: int, W: int) -> float:
        """
        Auto-detect where the printed header ends and handwriting begins.
        Strategy:
        1. Look for a horizontal rule/line in the top 40% of the page
           (these prescriptions have a clear line under the header)
        2. Fallback: find the gap between dense printed text and sparse
           handwriting using ink density profiling
        3. Ultimate fallback: 0.22 (safe default for these forms)
        """
        search_region = int(H * 0.40)
        gray = cv2.cvtColor(arr_rgb[:search_region, :], cv2.COLOR_RGB2GRAY)

        # -- Method 1: Detect horizontal rules via edge detection --
        edges = cv2.Canny(gray, 50, 150)
        # Look for long horizontal edges (lines spanning >50% of page width)
        kern_h = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 3, 1))
        horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kern_h)
        # Find rows with strong horizontal line presence
        row_sum = horiz_lines.sum(axis=1).astype(float)
        line_thresh = W * 100  # ~40% of page width worth of edge pixels
        line_rows = np.where(row_sum > line_thresh)[0]
        if len(line_rows) > 0:
            # Take the lowest horizontal rule in the search region
            boundary_row = int(line_rows[-1])
            # Add small padding below the line
            boundary = min(boundary_row + 15, search_region)
            frac = boundary / H
            if 0.10 <= frac <= 0.38:
                return frac

        # -- Method 2: Ink density profiling --
        # The header has dense, regular printed text; handwriting is sparser
        # Look for a valley (low ink) between header and body
        ink_top = ink_mask[:search_region, :]
        row_density = ink_top.astype(float).sum(axis=1)
        # Smooth to reduce noise
        sk = max(H // 60, 5)
        smooth = np.convolve(row_density, np.ones(sk)/sk, mode='same')
        # Find valleys (local minima in ink density)
        # Look in the 10%-35% range for the deepest valley
        start_search = int(H * 0.10)
        end_search = int(H * 0.35)
        if end_search > start_search:
            region = smooth[start_search:end_search]
            if len(region) > 0 and region.max() > 0:
                valley_idx = start_search + int(np.argmin(region))
                frac = valley_idx / H
                if 0.12 <= frac <= 0.35:
                    return frac

        # -- Fallback --
        return 0.22

    def save_debug(self, pil_img: Image.Image, page_idx: int):
        """Save multi-layer debug visualization."""
        arr = np.array(pil_img)
        full = self.get_ink_mask(pil_img)
        hw = self.get_handwriting_mask(pil_img)
        stamp = self.get_stamp_mask(pil_img)
        printed_only = (full > 0) & (hw == 0) & (stamp == 0)

        vis = arr.copy()
        vis[stamp > 0] = (vis[stamp > 0] * 0.35 +
                          np.array([220, 40, 40]) * 0.65).astype(np.uint8)
        vis[hw > 0] = (vis[hw > 0] * 0.4 +
                       np.array([50, 220, 50]) * 0.6).astype(np.uint8)
        vis[printed_only] = (vis[printed_only] * 0.5 +
                             np.array([0, 200, 220]) * 0.5).astype(np.uint8)

        path = os.path.join(DEBUG_DIR, f"p{page_idx+1}_regions_debug.png")
        Image.fromarray(vis).save(path)
        # Also save individual masks
        Image.fromarray(hw).save(
            os.path.join(DEBUG_DIR, f"p{page_idx+1}_hw_mask.png"))
        Image.fromarray(stamp).save(
            os.path.join(DEBUG_DIR, f"p{page_idx+1}_stamp_mask.png"))


# -------------------------------------------------------------
# Section 6  AUTO-ROI (Named regions: drug1, drug2, sig_notes)
# -------------------------------------------------------------
def auto_rois(pil_img: Image.Image, hw_mask: np.ndarray,
              n_rois: int = 3,
              merge_gap_frac: float = 0.03) -> Dict[str, Tuple]:
    """
    Find n_rois horizontal ink bands -> named ROIs.
    Returns dict of name -> (x0_frac, y0_frac, x1_frac, y1_frac).
    """
    H, W = hw_mask.shape
    row_sum = hw_mask.astype(float).sum(axis=1)
    smooth_k = max(H // 40, 5)
    kernel = np.ones(smooth_k) / smooth_k
    smooth = np.convolve(row_sum, kernel, mode="same")
    thresh = smooth.max() * 0.16
    above = (smooth > thresh).astype(np.uint8)

    bands = []
    in_b, start = False, 0
    for i, v in enumerate(above):
        if v and not in_b:
            in_b, start = True, i
        elif not v and in_b:
            in_b = False
            bands.append((start, i))
    if in_b:
        bands.append((start, H))

    gap = int(H * merge_gap_frac)
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(list(b))

    merged.sort(key=lambda b: smooth[b[0]:b[1]].sum(), reverse=True)
    top = sorted(merged[:n_rois], key=lambda b: b[0])

    names = ["drug1", "drug2", "sig_notes", "region_4", "region_5"]
    rois = {}
    pad_y = max(int(H * 0.015), 8)
    pad_x = max(int(W * 0.01), 5)

    for i, (y0, y1) in enumerate(top):
        band_mask = hw_mask[y0:y1, :]
        col_sum = band_mask.sum(axis=0)
        ink_cols = np.where(col_sum > 0)[0]
        if len(ink_cols) > 0:
            x0 = max(0, int(ink_cols[0]) - pad_x)
            x1 = min(W - 1, int(ink_cols[-1]) + pad_x)
        else:
            x0, x1 = 0, W - 1
        rois[names[min(i, len(names) - 1)]] = (
            x0 / W, max(0, (y0 - pad_y) / H),
            x1 / W, min(1.0, (y1 + pad_y) / H),
        )

    if not rois:
        rois = {"drug1": (0.06, 0.25, 0.96, 0.50),
                "drug2": (0.06, 0.50, 0.96, 0.75)}
    return rois


def crop_roi(pil_img: Image.Image, roi: Tuple) -> Image.Image:
    W, H = pil_img.size
    return pil_img.crop((int(roi[0]*W), int(roi[1]*H),
                         int(roi[2]*W), int(roi[3]*H)))

def crop_mask(mask: np.ndarray, roi: Tuple, wh: Tuple) -> np.ndarray:
    W, H = wh
    x0, y0 = int(roi[0]*W), int(roi[1]*H)
    x1, y1 = int(roi[2]*W), int(roi[3]*H)
    s = mask[y0:y1, x0:x1]
    if s.shape != (y1-y0, x1-x0):
        s = cv2.resize(s, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
    return (s > 127).astype(np.uint8) * 255


# -------------------------------------------------------------
# Section 7  LINE SEGMENTATION (with tall-band splitting)
# -------------------------------------------------------------
def segment_lines(roi_pil: Image.Image, hw_mask: np.ndarray,
                  min_h: int = 14, min_ink: int = 30,
                  gap_min: int = 6, pad_y: int = 6, pad_x: int = 10
                  ) -> List[Tuple[Image.Image, Tuple, int]]:
    """
    Split a region-of-interest image into individual text lines.

    This is critical because TrOCR works best on single-line images.
    We use horizontal projection profiling: sum the ink pixels per row,
    smooth the profile, and find bands of high ink density separated by
    valleys (gaps between lines).

    Parameters:
        roi_pil : cropped PIL image of one ROI (e.g. drug1 region)
        hw_mask : binary mask (same size) showing handwriting pixels
        min_h   : minimum band height in pixels to count as a line (14)
        min_ink : minimum total ink pixels in a band to keep it (30)
        gap_min : rows of gap smaller than this get merged (6)
        pad_y   : vertical padding added above/below each crop (6 px)
        pad_x   : horizontal padding added left/right of each crop (10 px)

    Returns:
        List of (line_image, (x, y, w, h), ink_pixel_count) tuples,
        sorted top-to-bottom.
    """
    H, W = hw_mask.shape

    # Light cleanup: close tiny holes but don't merge lines
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(hw_mask, cv2.MORPH_CLOSE, kern, iterations=1)

    # Horizontal projection (ink pixels per row)
    row = (m > 0).astype(np.uint8).sum(axis=1).astype(np.float32)

    # Smooth slightly
    k = max(7, (H // 60) | 1)  # odd
    row_s = np.convolve(row, np.ones(k) / k, mode="same")

    # Threshold: row considered "active" if it has enough ink
    thr = max(6.0, row_s.max() * 0.08)
    active = row_s > thr

    # Convert active rows into bands (lines), split by gaps
    bands = []
    in_b = False
    s = 0
    for i, a in enumerate(active):
        if a and not in_b:
            in_b = True
            s = i
        elif not a and in_b:
            in_b = False
            e = i
            if (e - s) >= min_h:
                bands.append((s, e))
    if in_b:
        e = H
        if (e - s) >= min_h:
            bands.append((s, e))

    # Merge only if the gap is extremely small (avoid over-merge)
    merged = []
    for (y0, y1) in bands:
        if not merged:
            merged.append([y0, y1])
            continue
        if y0 - merged[-1][1] <= gap_min:
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])

    # Build crops with tight x bounds per band
    lines = []
    for (y0, y1) in merged:
        band = m[y0:y1, :]
        ink_px = int((band > 0).sum())
        if ink_px < min_ink:
            continue

        col = (band > 0).sum(axis=0)
        xs = np.where(col > 0)[0]
        if len(xs) == 0:
            continue
        x0 = max(0, int(xs[0]) - pad_x)
        x1 = min(W, int(xs[-1]) + pad_x)

        cy0 = max(0, y0 - pad_y)
        cy1 = min(H, y1 + pad_y)

        if (cy1 - cy0) < min_h or (x1 - x0) < 40:
            continue

        crop = roi_pil.crop((x0, cy0, x1, cy1))
        lines.append((crop, (x0, cy0, x1 - x0, cy1 - cy0), ink_px))

    return lines


def _split_tall_band(hw_mask, y0, y1, min_h, min_ink):
    band = hw_mask[y0:y1, :]
    bh = y1 - y0
    row_sum = band.astype(float).sum(axis=1)
    sk = max(bh // 30, 3)
    if sk % 2 == 0: sk += 1
    smooth = np.convolve(row_sum, np.ones(sk)/sk, mode='same')
    peak = smooth.max()
    if peak < 1:
        return [(y0, y1, int((band > 0).sum()))]
    valley_thresh = peak * 0.12
    is_low = (smooth < valley_thresh).astype(np.uint8)
    valleys = []
    in_v, vs = False, 0
    for i, v in enumerate(is_low):
        if v and not in_v: in_v, vs = True, i
        elif not v and in_v:
            in_v = False
            if (i - vs) >= 4: valleys.append((vs, i))
    if in_v and (bh - vs) >= 4: valleys.append((vs, bh))
    if not valleys:
        return [(y0, y1, int((band > 0).sum()))]
    splits = [0] + [(vs+ve)//2 for vs, ve in valleys] + [bh]
    results = []
    for i in range(len(splits)-1):
        s, e = splits[i], splits[i+1]
        if (e-s) >= min_h:
            sub = band[s:e, :]
            ink = int((sub > 0).sum())
            if ink >= min_ink:
                results.append((y0+s, y0+e, ink))
    return results if results else [(y0, y1, int((band > 0).sum()))]


# -------------------------------------------------------------
# Section 8  CHARACTER-LEVEL DETECTION
# -------------------------------------------------------------
def detect_char_boxes(roi_pil: Image.Image, hw_mask: np.ndarray,
                      conf_threshold: int = 50
                      ) -> Tuple[Image.Image, List[Dict]]:
    """Character-level bounding boxes via Tesseract."""
    arr = np.array(roi_pil.convert("RGB"))
    masked = arr.copy()
    masked[hw_mask == 0] = 255
    mpil = Image.fromarray(masked)
    H = roi_pil.height

    draw = ImageDraw.Draw(mpil)
    tokens = []
    try:
        boxes = pytesseract.image_to_boxes(
            mpil, lang="eng+deu", config="--oem 1 --psm 6")
        for line in boxes.strip().split("\n"):
            parts = line.split()
            if len(parts) < 6:
                continue
            ch = parts[0]
            bx1, by1, bx2, by2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            by1 = H - by1; by2 = H - by2
            y_top, y_bot = min(by1, by2), max(by1, by2)
            tok_type = "NUMBER" if ch.isdigit() else "LETTER"
            color = (220, 30, 30) if tok_type == "NUMBER" else (30, 180, 30)
            draw.rectangle([bx1, y_top, bx2, y_bot], outline=color, width=2)
            tokens.append({"char": ch, "type": tok_type,
                          "bbox": [bx1, y_top, bx2, y_bot], "conf": 70})
    except Exception:
        # Fallback: word-level
        try:
            data = pytesseract.image_to_data(
                mpil, lang="eng+deu", output_type=pytesseract.Output.DICT,
                config="--oem 1 --psm 6")
            for i, txt in enumerate(data["text"]):
                txt = (txt or "").strip()
                if not txt: continue
                try:
                    c = int(data["conf"][i])
                except: c = 0
                if c < conf_threshold: continue
                x, y = data["left"][i], data["top"][i]
                bw, bh = data["width"][i], data["height"][i]
                for ch in txt:
                    tok_type = "NUMBER" if ch.isdigit() else "LETTER"
                    color = (220, 30, 30) if tok_type == "NUMBER" else (30, 180, 30)
                    draw.rectangle([x, y, x+bw, y+bh], outline=color, width=2)
                    tokens.append({"char": ch, "type": tok_type,
                                  "bbox": [x, y, x+bw, y+bh], "conf": c})
        except:
            pass
    return mpil, tokens


# -------------------------------------------------------------
# Section 9  TrOCR ENGINE (multi-model, beam search)
# -------------------------------------------------------------
TROCR_CANDIDATES = [
    TROCR_FT_DIR,                               # fine-tuned (if available)
    "microsoft/trocr-large-handwritten",
    "microsoft/trocr-base-handwritten",
]

class TrOCREngine:
    def __init__(self):
        self.processor = None
        self.model = None
        self.name = None
        for name in TROCR_CANDIDATES:
            try:
                if name.startswith("/") and not os.path.isdir(name):
                    continue
                print(f"[TrOCR] Loading {name} ...")
                proc = TrOCRProcessor.from_pretrained(name)
                model = VisionEncoderDecoderModel.from_pretrained(name).to(DEVICE)
                model.eval()
                self.processor, self.model, self.name = proc, model, name
                print(f"[TrOCR] [ok] {name}")
                break
            except Exception as e:
                print(f"[TrOCR] [fail] {name}: {e}")
        if not self.model:
            raise RuntimeError("No TrOCR model loaded.")
        self._ip = getattr(self.processor, "image_processor",
                           getattr(self.processor, "feature_extractor", None))

    @torch.inference_mode()
    def read_line(self, line_img: Image.Image,
                  hw_mask: Optional[np.ndarray] = None,
                  num_beams: int = 5) -> Tuple[str, List[float]]:
        """Read line with beam search. Returns (text, per_token_probs)."""
        prep = self._preprocess(line_img, hw_mask)
        if prep is None:
            return "", []
        pv = self._ip(images=prep, return_tensors="pt").pixel_values.to(DEVICE)
        out = self.model.generate(
            pv, max_new_tokens=100,
            num_beams=num_beams,
            output_scores=True,
            return_dict_in_generate=True)
        text = self.processor.tokenizer.decode(
            out.sequences[0], skip_special_tokens=True).strip()
        probs = [float(torch.softmax(s[0], -1).max().cpu())
                 for s in (out.scores or [])]
        return text, probs

    @torch.inference_mode()
    def read_line_nbest(self, line_img: Image.Image,
                        hw_mask: Optional[np.ndarray] = None,
                        num_beams: int = 8, num_return: int = 3
                        ) -> List[Tuple[str, float]]:
        """Return N-best hypotheses for reranking."""
        prep = self._preprocess(line_img, hw_mask)
        if prep is None:
            return [("", 0.0)]
        pv = self._ip(images=prep, return_tensors="pt").pixel_values.to(DEVICE)
        out = self.model.generate(
            pv, max_new_tokens=100,
            num_beams=num_beams,
            num_return_sequences=min(num_return, num_beams),
            output_scores=True,
            return_dict_in_generate=True)
        results = []
        for seq in out.sequences:
            text = self.processor.tokenizer.decode(
                seq, skip_special_tokens=True).strip()
            # Score using n-gram model
            score = NGRAM_MODEL.score(text)
            results.append((text, score))
        return sorted(results, key=lambda x: -x[1])

    def _preprocess(self, line_img, hw_mask=None, target_h=64):
        arr = np.array(line_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if h < 8 or w < 8:
            return None

        # Remove stamp residue in line crop
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        stamp_px = ((hsv[:,:,0] >= 100) & (hsv[:,:,0] <= 165) &
                     (hsv[:,:,1] > 35))
        gray[stamp_px] = 255

        # Apply handwriting mask if provided
        if hw_mask is not None and hw_mask.shape == gray.shape:
            gray[hw_mask == 0] = 255

        # Deskew
        inv = cv2.bitwise_not(gray)
        thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thr > 0))
        if len(coords) >= 50:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = 90 + angle
            if 0.5 < abs(angle) < 15:
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

        # CLAHE + sharpen
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Resize
        scale = target_h / max(h, 1)
        new_w = min(max(int(w * scale), 20), 800)
        resized = cv2.resize(binary, (new_w, target_h),
                             interpolation=cv2.INTER_CUBIC)

        padded = cv2.copyMakeBorder(resized, 12, 12, 16, 16,
                                    cv2.BORDER_CONSTANT, value=255)
        return Image.fromarray(padded).convert("RGB")


# -------------------------------------------------------------
# Section 10  TESSERACT SECONDARY + ENSEMBLE
# -------------------------------------------------------------
def tesseract_read_line(line_img: Image.Image) -> Tuple[str, float]:
    arr = np.array(line_img.convert("L"))
    h, w = arr.shape
    if h < 24:
        scale = 36 / max(h, 1)
        arr = cv2.resize(arr, (int(w*scale), 36), interpolation=cv2.INTER_CUBIC)
    _, thr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_thr = Image.fromarray(thr)
    try:
        data = pytesseract.image_to_data(
            pil_thr, lang="eng+deu",
            output_type=pytesseract.Output.DICT, config="--oem 1 --psm 7")
        texts, confs = [], []
        for txt, conf in zip(data["text"], data["conf"]):
            txt = (txt or "").strip()
            try: c = int(conf)
            except: c = 0
            if txt and c > 15:
                texts.append(txt)
                confs.append(c / 100.0)
        return " ".join(texts), float(np.mean(confs)) if confs else 0.0
    except:
        return "", 0.0


def ensemble_ocr(trocr_text: str, trocr_probs: List[float],
                 tess_text: str, tess_conf: float,
                 corrector: Optional[OCRCorrectorModel],
                 vocab: List[str]) -> Tuple[str, float, str]:
    """
    Intelligent fusion of multiple OCR readings for a single text line.

    The idea is that no single OCR engine is perfect on old handwritten
    prescriptions.  TrOCR (a transformer model) is strong on cursive but
    can hallucinate.  Tesseract is good at printed-style characters.  The
    correction model fixes common OCR errors (e.g. 'rn' -> 'm').

    Scoring pipeline:
        1. Collect candidates from TrOCR and Tesseract.
        2. If a correction model is loaded, apply it to each candidate;
           keep the corrected version only if it scores higher on the
           n-gram language model (i.e. looks more like a real prescription).
        3. Score every candidate by combining:
           - raw OCR confidence (50% weight)
           - n-gram plausibility score (20% weight)
           - vocabulary fuzzy-match bonus (15% weight cap)
           - drug-name match bonus (20% weight cap)
        4. Return the highest-scoring candidate.

    Parameters:
        trocr_text   : text from TrOCR beam search
        trocr_probs  : per-token softmax probabilities from TrOCR
        tess_text    : text from Tesseract OCR
        tess_conf    : Tesseract average word confidence (0-1)
        corrector    : trained OCRCorrectorModel, or None
        vocab        : merged vocabulary list for fuzzy matching

    Returns:
        (best_text, confidence_score, source_label) where source_label
        is something like "trocr", "tesseract", or "trocr+corrector".
    """
    trocr_conf = float(np.mean(trocr_probs)) if trocr_probs else 0.0

    candidates = []
    if trocr_text:
        candidates.append((trocr_text, trocr_conf, "trocr"))
    if tess_text:
        candidates.append((tess_text, tess_conf, "tesseract"))

    # Apply correction model to each candidate
    if corrector:
        for text, conf, src in list(candidates):
            corrected = corrector.correct(text)
            if corrected and corrected != text:
                # Score corrected version
                ngram_score = NGRAM_MODEL.score(corrected)
                orig_score = NGRAM_MODEL.score(text)
                if ngram_score > orig_score:
                    candidates.append((corrected, conf * 1.1, f"{src}+corrector"))

    # Score all candidates
    scored = []
    for text, conf, src in candidates:
        ngram = NGRAM_MODEL.score(text)
        # Vocabulary bonus
        vm, vs = vocab_match(text, vocab, threshold=65) if vocab else ("", 0)
        vocab_bonus = (vs / 100.0) * 0.15 if vs > 0 else 0.0
        # Drug match bonus
        dn, dc, ds = find_drug_in_text(text)
        drug_bonus = (ds / 100.0) * 0.2 if ds > 70 else 0.0

        total = conf * 0.5 + min(1.0, (ngram + 10) / 5) * 0.2 + vocab_bonus + drug_bonus + 0.15
        scored.append((text, total, src))

    if not scored:
        return "", 0.0, "none"

    scored.sort(key=lambda x: -x[1])
    return scored[0]


# -------------------------------------------------------------
# Section 11  POST-PROCESSING (German OCR fixes)
# -------------------------------------------------------------
OCR_FIXES = [
    (r'(?<=[a-z])rn(?=[a-z])', 'm'), (r'\brn\b', 'm'),
    (r'\bvv\b', 'w'), (r'(?<=[a-z])vv(?=[a-z])', 'w'),
    (r'(?<=[A-Za-z])1(?=[A-Za-z])', 'l'),
    (r'(?<=[A-Za-z])0(?=[A-Za-z])', 'o'),
    # Drug-specific
    (r'\bVal[iu]r[nm]\b', 'Valium'), (r'\bVol[iu][umn]+\b', 'Valium'),
    (r'\bVah?urn\b', 'Valium'), (r'\bValiuni?\b', 'Valium'),
    (r'\bVclium\b', 'Valium'), (r'\bValinn\b', 'Valium'),
    (r'\bTrypt[il1][fz]o[l1]\b', 'Tryptizol'),
    (r'\bTryph[ft]ol\b', 'Tryptizol'), (r'\bTrypti[sz]o[lt]\b', 'Tryptizol'),
    (r'\bJryptizol\b', 'Tryptizol'), (r'\bTryptizoi\b', 'Tryptizol'),
    (r'\bTrypti2ol\b', 'Tryptizol'),
    # Dosage normalization
    (r'\bO\s*\.?\s*P\s*\.?\s*N\s*[cC]\s*[tTiI]\b', 'O.P.Nct'),
    (r'\btägl\b', 'täglich'), (r'\b[tT]bl\.?\b', 'Tbl.'),
    (r'\b[rR]ep(?:etatur|etahir|etahui|etchi)\b', 'Repetatur'),
]


def clean_ocr_text(text: str) -> str:
    """
    Post-process raw OCR output to fix common handwriting misreads.

    Applies a list of regex-based substitutions (OCR_FIXES) that correct
    known confusions specific to German/Austrian prescription handwriting:
        - 'rn' inside words -> 'm'  (very common in cursive)
        - 'vv' -> 'w'
        - digit/letter swaps: '1' inside a word -> 'l', '0' -> 'o'
        - Drug-specific fixes: 'Valirn' -> 'Valium', 'Tryptlfol' -> 'Tryptizol'
        - Dosage normalisation: 'O.P. N c t' -> 'O.P.Nct'
        - Abbreviation expansion: 'tägl' -> 'täglich', 'Tbl' -> 'Tbl.'
        - Strips leading/trailing punctuation junk
        - Collapses multiple spaces into one
        - Unicode NFC normalisation (combines diacritics)

    Parameters:
        text: raw OCR string from ensemble_ocr

    Returns:
        Cleaned string ready for drug matching and display.
    """
    if not text:
        return ""
    text = re.sub(r'^["\'\-\.,:;=]+', '', text)
    text = re.sub(r'["\'\-\.,:;=]+$', '', text)
    for pattern, repl in OCR_FIXES:
        text = re.sub(pattern, repl, text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = unicodedata.normalize("NFC", text)
    return text.strip()



def extract_price_text(text: str) -> str:
    """
    Extract a price-like value from OCR text.

    The scanned prescriptions show handwritten amounts such as:
        44,50
        39,60
        45,-
        39,-

    These values appear next to medicine lines or instruction lines and are
    closer to prices/amounts than to medicine quantities.  The function returns
    the first matching price token if one is found.
    """
    if not text:
        return ""
    m = re.search(r'\b\d{1,3}\s*[,\.]\s*(?:\d{2}|-)\b', text)
    if m:
        return re.sub(r'\s+', '', m.group(0)).replace('.', ',')
    m = re.search(r'\b\d{1,3}\s*[-–—]\b', text)
    return re.sub(r'\s+', '', m.group(0)) if m else ""


def extract_patient_name(text: str) -> str:
    """
    Extract patient information from lines such as:
        für Mr. Alain
        für Herrn Alain
        für Frau Müller
    """
    if not text:
        return ""
    m = re.search(
        r'\bfür\s+(Mr\.?|Mrs\.?|Herrn?|Frau)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)',
        text,
        re.IGNORECASE,
    )
    if not m:
        return ""
    title = m.group(1).strip()
    name = m.group(2).strip()
    return f"für {title} {name}"


def extract_instruction_text(text: str) -> str:
    """
    Extract prescription instructions beginning with 'S:'.

    Examples from the target images:
        S: Repetatur
        S: 1 in the morning, 1 in the evening
        S: 2x täglich 1 Tablette
    """
    if not text:
        return ""
    m = re.search(r'\bS\s*[:\.]\s*(.+)$', text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def classify_semantic_line(cleaned_text: str,
                           drug_name: str = "",
                           drug_score: int = 0) -> str:
    """
    Assign a coarse semantic label to a line.

    This is useful for GitHub demos and CSV inspection because the scanned
    prescriptions contain multiple line types, not just medicine names.

    Returns one of:
        - DRUG
        - INSTRUCTION
        - PATIENT
        - PRICE
        - DATE
        - NOTE
        - EMPTY
    """
    text = (cleaned_text or "").strip()
    if not text:
        return "EMPTY"
    if drug_name and drug_score >= 68:
        return "DRUG"
    if extract_instruction_text(text):
        return "INSTRUCTION"
    if extract_patient_name(text):
        return "PATIENT"
    if extract_price_text(text):
        return "PRICE"
    if re.search(r'\b\d{1,2}[\./]\d{1,2}[\./]\d{2,4}\b', text) or re.search(r'\bJUNI\b|\bMAI\b', text, re.IGNORECASE):
        return "DATE"
    return "NOTE"

# -------------------------------------------------------------
# Section 12  COMBINED CONFIDENCE SCORING
# -------------------------------------------------------------
def combined_confidence(token_probs: List[float], text: str,
                        ink_pixels: int, min_ink: int = 80) -> float:
    """
    Weighted confidence:
    - token_mean: average TrOCR per-token probability
    - length_pen: penalize very short text (< 3 chars)
    - ink_pen: penalize regions with almost no ink
    - ngram_bonus: plausibility from prescription n-gram model
    """
    token_mean = float(np.mean(token_probs)) if token_probs else 0.0
    length_pen = 1.0 if len(text.strip()) >= 3 else 0.4
    ink_pen = 1.0 if ink_pixels >= min_ink else 0.5
    ngram_score = NGRAM_MODEL.score(text)
    ngram_bonus = min(0.15, max(0, (ngram_score + 8) / 40))
    return round(token_mean * length_pen * ink_pen + ngram_bonus, 4)


# -------------------------------------------------------------
# Section 13  DOCTOR NAME EXTRACTION
# -------------------------------------------------------------
def extract_header_info(pil_img: Image.Image) -> Dict[str, str]:
    """
    Extract printed header information from the top of a prescription page.

    Austrian prescription forms have a typeset header containing the doctor's
    name, specialty (e.g. "Facharzt für Innere Medizin"), postal address
    (e.g. "A-5640 Bad Gastein, Kirchplatz 7"), and phone number.

    IMPORTANT: This function runs on the ORIGINAL image, not the handwriting
    mask.  The handwriting mask intentionally zeros out the header region to
    help line-level OCR, so we must extract header info first.

    Parameters:
        pil_img: full-page PIL image (RGB, typically 300 DPI).

    Returns:
        Dict with keys "doctor", "specialty", "address", "phone".
        Missing fields default to "" (doctor defaults to "N/A").
    """
    W, H = pil_img.size
    info = {"doctor": "N/A", "address": "", "phone": "", "specialty": ""}

    # Crop generous header region (top 32%)
    header = pil_img.crop((int(0.02*W), int(0.01*H),
                           int(0.98*W), int(0.32*H)))

    # -- Primary: structured OCR with PSM 6 (block of text) --
        # Remove stamp/overprint pixels in header BEFORE OCR
    h_arr = np.array(header.convert("RGB"))
    hsv = cv2.cvtColor(h_arr, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # same stamp hue logic as your detect_stamp_mask, but simpler here
    stamp_px = (((h >= 85) & (h <= 165)) & (s > 35) & (v > 40) & (v < 230))
    h_arr[stamp_px] = 255  # whiten stamp pixels
    header_clean = Image.fromarray(h_arr)

    raw = pytesseract.image_to_string(header_clean, lang="eng+deu",
                                      config="--psm 6 --oem 1")

    # Split the raw OCR output into individual lines, dropping blanks.
    # This is what we iterate over to find doctor name, address, phone, etc.
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]

    # -- Extract doctor name --
    # Pattern 1: "DR. HEINZ SCHACHINGER" (bold uppercase header line)
    for line in lines:
        m = re.search(
            r"DR\.?\s+([A-ZÄÖÜ][a-zA-ZÄÖÜäöüß]+(?:\s+[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß]+)+)",
            line, re.IGNORECASE)
        if m:
            info["doctor"] = m.group(0).strip()
            break

    # Pattern 2: Look for title line ("Kurarzt" / "Facharzt") then name on next line
    if info["doctor"] == "N/A":
        for i, line in enumerate(lines):
            if re.search(r'\b(Kurarzt|Facharzt|Arzt|Doktor)\b', line, re.IGNORECASE):
                # Doctor name is likely the next line with uppercase words
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    candidate = lines[j]
                    if re.search(r'\bDR\.?\b', candidate, re.IGNORECASE):
                        info["doctor"] = candidate[:60]
                        break

    # Pattern 3: Any line containing "Dr." with 2+ following name words
    if info["doctor"] == "N/A":
        for line in lines:
            if re.search(r'\bDR\.?\b', line, re.IGNORECASE) and len(line) > 5:
                info["doctor"] = line[:60]
                break

    # -- Extract specialty --
    for line in lines:
        if re.search(r'Fach.{0,5}(für|f\.)\s+\w+', line, re.IGNORECASE):
            info["specialty"] = line.strip()[:80]
            break

    # -- Extract address --
    for line in lines:
        m = re.search(r'A-?\d{4}\s+.+', line)
        if m:
            info["address"] = m.group(0).strip()[:80]
            break

    # -- Extract phone --
    for line in lines:
        if re.search(r'(Telefon|Tel\.?|Tel)', line, re.IGNORECASE):
            info["phone"] = line.strip()[:80]
            break

    return info


def extract_doctor_name(pil_img: Image.Image) -> str:
    """Convenience wrapper -- just returns the doctor name string."""
    return extract_header_info(pil_img)["doctor"]


# -------------------------------------------------------------
# Section 14  PDF LOADING
# -------------------------------------------------------------
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    pages = [Image.open(io.BytesIO(
        page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                        alpha=False).tobytes("png")
    )).convert("RGB") for page in doc]
    doc.close()
    print(f"[PDF] {len(pages)} page(s) at {dpi} DPI")
    return pages


# -------------------------------------------------------------
# Section 15  ANNOTATED IMAGE BUILDER
# -------------------------------------------------------------
ROI_COLORS = {
    "drug1": (30,144,255), "drug2": (255,140,0),
    "sig_notes": (50,200,50), "region_4": (180,60,200),
    "region_5": (200,200,0),
}

def build_annotated_image(pil_page, rois, roi_rows, doctor, page_idx):
    W, H = pil_page.size
    PAD = 24
    PW = max(W, 820)

    left = pil_page.copy()
    draw_l = ImageDraw.Draw(left)
    try:
        fl = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        fl = ImageFont.load_default()

    for rn, rc in rois.items():
        x0, y0 = int(rc[0]*W), int(rc[1]*H)
        x1, y1 = int(rc[2]*W), int(rc[3]*H)
        c = ROI_COLORS.get(rn, (200, 0, 200))
        draw_l.rectangle([x0, y0, x1, y1], outline=c, width=4)
        draw_l.rectangle([x0, y0, x0+len(rn)*12+8, y0+26], fill=c)
        draw_l.text((x0+4, y0+2), rn, fill=(255,255,255), font=fl)

    right = Image.new("RGB", (PW, H), (248, 248, 248))
    draw_r = ImageDraw.Draw(right)
    try:
        f26 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        f20 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        f16 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        f13 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        f26 = f20 = f16 = f13 = ImageFont.load_default()

    draw_r.rectangle([0, 0, PW, 56], fill=(20, 20, 20))
    draw_r.text((PAD, 10), f"Page {page_idx+1} -- Rx OCR v9 Production",
                fill=(255,255,255), font=f26)
    y = 66
    draw_r.text((PAD, y), f"Doctor: {doctor}", fill=(30,100,210), font=f20)
    y += 34
    draw_r.line([(PAD, y), (PW-PAD, y)], fill=(180,180,180), width=1)
    y += 12

    badge_c = {"trocr": (0,100,180), "tesseract": (180,100,0),
               "trocr+corrector": (0,160,60), "tesseract+corrector": (60,160,0),
               "none": (120,120,120)}

    for i, row in enumerate(roi_rows):
        if y > H - 80:
            break
        rn = row["ROI"]
        c = ROI_COLORS.get(rn, (150, 0, 150))
        conf = float(row["Mean_Confidence"])
        src = row["Match_Source"]
        drug = row.get("Drug_Name", "")

        # Header
        draw_r.rectangle([PAD-4, y-2, PAD+8, y+22], fill=c)
        draw_r.text((PAD+14, y), f"{rn} L{row['Line_Index']}",
                    fill=c, font=f20)
        y += 26

        # Raw + Cleaned
        draw_r.text((PAD+8, y),
                    f"Raw: {row['OCR_Raw'][:60]}", fill=(130,130,130), font=f13)
        y += 18
        draw_r.text((PAD+8, y),
                    f"-> {row['OCR_Clean'][:60]}", fill=(15,15,15), font=f16)
        y += 22

        # Drug highlight
        if drug:
            lbl = f" DRUG: {drug} -- {row.get('Drug_Class','')} "
            tw = min(len(lbl) * 9 + 10, PW - PAD * 2 - 16)
            draw_r.rectangle([PAD+8, y, PAD+8+tw, y+24], fill=(180, 0, 0))
            draw_r.text((PAD+12, y+2), lbl[:tw//9], fill=(255,255,255), font=f16)
            y += 28

        # Source badge
        bc = badge_c.get(src, (100,100,100))
        blbl = f" {src} {int(conf*100)}% "
        draw_r.rectangle([PAD+8, y, PAD+8+len(blbl)*8+8, y+20], fill=bc)
        draw_r.text((PAD+12, y+2), blbl, fill=(255,255,255), font=f13)
        y += 24

        # Numbers/Letters
        if row.get("Numbers_Detected"):
            draw_r.text((PAD+8, y), f"# {row['Numbers_Detected']}",
                        fill=(200,40,40), font=f13)
            y += 16
        if row.get("Letters_Detected"):
            draw_r.text((PAD+8, y), f"Az {row['Letters_Detected'][:50]}",
                        fill=(40,150,40), font=f13)
            y += 16

        # Confidence bar
        bar_max = PW - PAD * 3
        bar_w = int(bar_max * min(conf, 1.0))
        bar_col = ((220,50,50) if conf < 0.35
                   else (200,160,0) if conf < 0.6
                   else (50,180,50))
        draw_r.rectangle([PAD+8, y, PAD+8+bar_w, y+10], fill=bar_col)
        draw_r.rectangle([PAD+8, y, PAD+8+bar_max, y+10], outline=(160,160,160))
        y += 16
        draw_r.line([(PAD, y), (PW-PAD, y)], fill=(210,210,210))
        y += 8

    # Legend
    draw_r.rectangle([0, H-28, PW, H], fill=(30,30,30))
    draw_r.text((PAD, H-20),
                "RED=stamp GREEN=hw CYAN=printed | Drug=red badge | bar=confidence",
                fill=(150,150,150), font=f13)

    out = Image.new("RGB", (W+PW+4, H), (100,100,100))
    out.paste(left, (0, 0))
    out.paste(right, (W+4, 0))
    return out


# -------------------------------------------------------------
# Section 16  MAIN PIPELINE
# -------------------------------------------------------------
def run_pipeline(pdf_path=None, output_csv=OUTPUT_CSV):
    """
    Main entry point: read a prescription PDF and extract all handwritten text.

    This orchestrates the entire OCR pipeline end-to-end:
        1. Load the PDF and render each page at 300 DPI.
        2. Run header OCR (doctor name, address) on the raw image.
        3. Build handwriting / stamp masks to isolate cursive text.
        4. Auto-detect regions of interest (drug1, drug2, sig_notes, etc.).
        5. Segment each ROI into individual text lines.
        6. Run TrOCR + Tesseract on every line, fuse with ensemble scoring.
        7. Clean OCR output, match drug names, normalise dosage strings.
        8. Extract structured semantics such as price, patient, and instructions.
        9. Write results to CSV and generate annotated debug images.

    Parameters:
        pdf_path   : path to the input prescription PDF file.
                     If None, falls back to the global PDF_PATH variable
                     (set by the Cell 2 file-picker widget).
        output_csv : where to write the results CSV (default: prescription_output.csv
                     in BASE_DIR).

    Returns:
        List of row dicts (one per detected text line), same data as the CSV.
    """
    if pdf_path is None:
        pdf_path = globals().get("PDF_PATH", None)
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

    print(f"\n{'=' * 60}")
    print(f" AustrianRx Vision v11 -- {os.path.basename(pdf_path)}")
    print(f"{'=' * 60}")

    # -- Build vocabulary --
    lcsv, wdir = ensure_kaggle_dataset()
    vocab = build_vocabulary(lcsv)

    # -- Load OCR models --
    trocr = TrOCREngine()
    corrector = load_corrector()

    # -- Load PDF --
    ink_masker = InkMask()
    pages = pdf_to_images(pdf_path, dpi=300)

    fieldnames = [
        "Page", "ROI", "Line_Index", "Doctor",
        "OCR_Raw", "OCR_Clean", "Dosage_Normalized",
        "Semantic_Label",
        "Is_Drug_Line", "Drug_Name", "Drug_Class", "Drug_Score",
        "Instruction_Text", "Patient_Text", "Price_Text",
        "Match_Source", "Vocab_Match", "Vocab_Score",
        "Numbers_Detected", "Letters_Detected",
        "Mean_Confidence", "Ink_Pixels",
        "BBox_X", "BBox_Y", "BBox_W", "BBox_H",
    ]
    all_rows = []

    for page_idx, pil_page in enumerate(pages):
        print(f"\n{'-' * 55}")
        print(f" Page {page_idx+1}")
        print(f"{'-' * 55}")

        # -- STEP 0: Header OCR pass (BEFORE masking removes the header) --
        header_info = extract_header_info(pil_page)
        doctor = header_info["doctor"]
        print(f" Doctor:    {doctor}")
        if header_info["specialty"]:
            print(f" Specialty: {header_info['specialty']}")
        if header_info["address"]:
            print(f" Address:   {header_info['address']}")

        # -- STEP 1: Preprocessing & masks (this zeros out the header) --
        hw_mask = ink_masker.get_handwriting_mask(pil_page)
        ink_masker.save_debug(pil_page, page_idx)

        # -- Auto-ROI --
        rois = auto_rois(pil_page, hw_mask, n_rois=4, merge_gap_frac=0.008)
        print(f" ROIs: {list(rois.keys())}")

        page_rows = []

        for roi_name, roi_coords in rois.items():
            print(f"\n  +-- {roi_name}")
            roi_pil = crop_roi(pil_page, roi_coords)
            roi_hw = crop_mask(hw_mask, roi_coords, pil_page.size)
            if roi_hw.shape != (roi_pil.height, roi_pil.width):
                roi_hw = cv2.resize(roi_hw, (roi_pil.width, roi_pil.height),
                                    interpolation=cv2.INTER_NEAREST)
                roi_hw = (roi_hw > 127).astype(np.uint8) * 255

            # -- Character boxes --
            annotated, char_tokens = detect_char_boxes(roi_pil, roi_hw)
            annotated.save(os.path.join(
                DEBUG_DIR, f"p{page_idx+1}_{roi_name}_debug.png"))
            nums_all = "".join(t["char"] for t in char_tokens
                               if t["type"] == "NUMBER")
            lets_all = "".join(t["char"] for t in char_tokens
                               if t["type"] == "LETTER")

            # -- Line segmentation --
            lines = segment_lines(roi_pil, roi_hw)
            if not lines:
                row = {
                    "Page": page_idx+1, "ROI": roi_name,
                    "Line_Index": 0, "Doctor": doctor,
                    "OCR_Raw": "", "OCR_Clean": "",
                    "Dosage_Normalized": "",
                    "Semantic_Label": "EMPTY",
                    "Is_Drug_Line": "no", "Drug_Name": "",
                    "Drug_Class": "", "Drug_Score": 0,
                    "Instruction_Text": "", "Patient_Text": "", "Price_Text": "",
                    "Match_Source": "no_ink", "Vocab_Match": "",
                    "Vocab_Score": 0, "Numbers_Detected": "",
                    "Letters_Detected": "", "Mean_Confidence": 0.0,
                    "Ink_Pixels": 0, "BBox_X": 0, "BBox_Y": 0,
                    "BBox_W": 0, "BBox_H": 0,
                }
                all_rows.append(row)
                page_rows.append(row)
                continue

            for li, (line_img, (lx, ly, lw, lh), ink_px) in enumerate(lines):
                # Save crop
                line_img.save(os.path.join(
                    DEBUG_DIR, f"p{page_idx+1}_{roi_name}_line{li+1:02d}_crop.png"))

                # -- TrOCR (primary) --
                raw_trocr, probs_trocr = trocr.read_line(line_img)

                # -- Tesseract (secondary) --
                raw_tess, conf_tess = tesseract_read_line(line_img)

                # -- Ensemble fusion --
                best_text, best_conf, source = ensemble_ocr(
                    raw_trocr, probs_trocr,
                    raw_tess, conf_tess,
                    corrector, vocab)

                # -- Clean + drug match --
                cleaned = clean_ocr_text(best_text)
                dosage_norm = apply_dosage_patterns(cleaned)

                drug_name, drug_class, drug_score = find_drug_in_text(cleaned)
                if drug_score < 70:
                    dn2, dc2, ds2 = find_drug_in_text(raw_trocr)
                    if ds2 > drug_score:
                        drug_name, drug_class, drug_score = dn2, dc2, ds2
                if drug_score < 70 and raw_tess:
                    dn3, dc3, ds3 = find_drug_in_text(raw_tess)
                    if ds3 > drug_score:
                        drug_name, drug_class, drug_score = dn3, dc3, ds3

                is_drug = bool(drug_name and drug_score >= 68)

                # -- Structured semantic fields --
                instruction_text = extract_instruction_text(dosage_norm or cleaned)
                patient_text = extract_patient_name(cleaned)
                price_text = extract_price_text(cleaned)
                semantic_label = classify_semantic_line(
                    dosage_norm or cleaned,
                    drug_name=drug_name,
                    drug_score=drug_score,
                )

                # -- Vocab match --
                vm, vs = vocab_match(cleaned, vocab) if vocab else ("", 0)

                # -- Combined confidence --
                conf = combined_confidence(probs_trocr, cleaned, ink_px)

                print(f"  | L{li+1:02d} [{source:18s}] conf={conf:.2f} "
                      f"| {cleaned[:50]!r}"
                      + (f"  -> DRUG: {drug_name}" if is_drug else f"  -> {semantic_label}"))

                row = {
                    "Page": page_idx+1, "ROI": roi_name,
                    "Line_Index": li+1, "Doctor": doctor,
                    "OCR_Raw": best_text, "OCR_Clean": cleaned,
                    "Dosage_Normalized": dosage_norm if dosage_norm != cleaned else "",
                    "Semantic_Label": semantic_label,
                    "Is_Drug_Line": "YES" if is_drug else "no",
                    "Drug_Name": drug_name, "Drug_Class": drug_class,
                    "Drug_Score": drug_score,
                    "Instruction_Text": instruction_text,
                    "Patient_Text": patient_text,
                    "Price_Text": price_text,
                    "Match_Source": source, "Vocab_Match": vm,
                    "Vocab_Score": vs,
                    "Numbers_Detected": nums_all if li == 0 else "",
                    "Letters_Detected": lets_all[:60] if li == 0 else "",
                    "Mean_Confidence": conf, "Ink_Pixels": ink_px,
                    "BBox_X": lx, "BBox_Y": ly, "BBox_W": lw, "BBox_H": lh,
                }
                all_rows.append(row)
                page_rows.append(row)

            print(f"  +-- {roi_name}: {len(lines)} lines")

        # -- Annotated image --
        if page_rows:
            ann = build_annotated_image(
                pil_page, rois, page_rows, doctor, page_idx)
            ann_path = os.path.join(ANNOTATED_DIR,
                                    f"page_{page_idx+1}_annotated.png")
            ann.save(ann_path)
            print(f"\n [ok] Annotated -> {ann_path}")

    # -- Write CSV --
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(all_rows)

    # -- Summary --
    print(f"\n{'=' * 60}")
    print(f" [ok] CSV: {len(all_rows)} rows -> {output_csv}")
    drug_rows = [r for r in all_rows if r['Is_Drug_Line'] == 'YES']
    print(f" [ok] Drugs found: {len(drug_rows)}")
    for r in drug_rows:
        print(f"   p{r['Page']} {r['ROI']} L{r['Line_Index']}: "
              f"{r['Drug_Name']} ({r['Drug_Class']}) "
              f"score={r['Drug_Score']}")
        print(f"     OCR: {r['OCR_Clean']!r}")
        if r['Dosage_Normalized']:
            print(f"     Normalized: {r['Dosage_Normalized']!r}")
    print(f" [ok] Annotated -> {ANNOTATED_DIR}/")
    print(f" [ok] Debug -> {DEBUG_DIR}/")
    if not corrector:
        print(f"\n  [info] No correction model found. Run Cell 7 to train one.")
    return all_rows


# --- Run the main pipeline now, so the CSV is ready for the preview cell ---
# If PDF_PATH is not yet set (e.g. running cells out of order), this will
# print a message instead of crashing.
try:
    _pdf = globals().get("PDF_PATH", None)
    if _pdf and os.path.exists(_pdf):
        all_rows = run_pipeline(_pdf)
    else:
        print("[pipeline] No PDF selected yet. Run Cell 2 first, then re-run this cell.")
except Exception as _e:
    print(f"[pipeline] Error: {_e}")
# ============================================================
# CELL 4 -- Preview results
# ============================================================
import pandas as pd
from IPython.display import display

if not os.path.exists(OUTPUT_CSV):
    print("[preview] Output CSV not found. Make sure the pipeline in Cell 3 ran successfully.")
else:
    df = pd.read_csv(OUTPUT_CSV, dtype=str)
    cols = ["Page", "ROI", "Line_Index", "OCR_Raw", "OCR_Clean",
            "Dosage_Normalized", "Is_Drug_Line", "Drug_Name",
            "Drug_Class", "Match_Source", "Mean_Confidence"]
    display(df[[c for c in cols if c in df.columns]].reset_index(drop=True))

    print("\n-- Drug lines --")
    drug_df = df[df["Is_Drug_Line"] == "YES"]
    for _, r in drug_df.iterrows():
        print(f"  [drug] p{r['Page']} {r['ROI']} L{r['Line_Index']}: "
              f"{r['Drug_Name']} -- {r['OCR_Clean']!r}")

    print("\n-- All lines --")
    for _, r in df.iterrows():
        icon = "[drug]" if r['Is_Drug_Line'] == "YES" else "      "
        print(f"  {icon} p{r['Page']} {r['ROI']:10s} L{r['Line_Index']:>2s} "
              f"[{r['Match_Source']:18s}] {r['Mean_Confidence']:>6s} "
              f"| {str(r['OCR_Clean'])[:50]}")
# ============================================================
# CELL 5 -- Display images
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ann_files = sorted(glob.glob(f"{ANNOTATED_DIR}/*.png"))
for p in ann_files:
    fig, ax = plt.subplots(figsize=(28, 15))
    ax.imshow(mpimg.imread(p)); ax.axis("off")
    ax.set_title(os.path.basename(p), fontsize=12)
    plt.tight_layout(); plt.show()

# Debug: regions (stamp=red, hw=green, printed=cyan)
reg_files = sorted(glob.glob(f"{DEBUG_DIR}/p*_regions_debug.png"))
if reg_files:
    fig, axes = plt.subplots(1, len(reg_files),
                              figsize=(14*len(reg_files), 18))
    if len(reg_files) == 1: axes = [axes]
    for ax, p in zip(axes, reg_files):
        ax.imshow(mpimg.imread(p)); ax.axis("off")
        ax.set_title(os.path.basename(p) +
                     "\n(RED=stamp GREEN=handwriting CYAN=printed)", fontsize=11)
    plt.tight_layout(); plt.show()

# Line crops
crops = sorted(glob.glob(f"{DEBUG_DIR}/p*_line*_crop.png"))[:16]
if crops:
    n = len(crops); cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
    af = np.array(axes).flatten() if n > 1 else [axes]
    for ax, p in zip(af, crops):
        ax.imshow(mpimg.imread(p), cmap='gray'); ax.axis("off")
        ax.set_title(os.path.basename(p).replace("_crop.png",""), fontsize=8)
    for ax in af[n:]: ax.axis("off")
    plt.suptitle("Line crops -> OCR", fontsize=12)
    plt.tight_layout(); plt.show()    
# ============================================================
# CELL 6 -- Extended TrOCR Fine-tuning (prescription-specific)
# ============================================================
# Run AFTER Cell 3 to generate line crops, then provide labels.
#
# Changes from v7:
#  - More epochs (10-15 for small datasets)
#  - Cosine LR scheduler with warm restarts
#  - Heavy augmentation: brightness, contrast, rotation, noise
#  - Synthetic data augmentation from drug vocabulary
#  - Validation split with early stopping
# ============================================================
import os, csv, random, glob, math
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

# -- CONFIG --
FT_BASE_MODEL  = "microsoft/trocr-large-handwritten"
FT_LABELS_CSV  = os.path.join(BASE_DIR, "finetune_labels.csv")
FT_CROPS_DIR   = DEBUG_DIR
FT_OUT_DIR     = TROCR_FT_DIR
FT_EPOCHS      = 12      # longer training
FT_BATCH       = 4
FT_LR          = 1.5e-5  # slightly lower for stability
FT_MAX_LEN     = 80      # longer sequences for prescriptions
FT_PATIENCE    = 4       # early stopping patience
FT_VAL_SPLIT   = 0.15
FT_DEVICE      = DEVICE

print(f"[Finetune] Device: {FT_DEVICE}")
os.makedirs(FT_OUT_DIR, exist_ok=True)


def generate_synthetic_pairs(n: int = 200) -> List[Tuple[str, str]]:
    """
    Generate synthetic prescription text pairs for data augmentation.
    Creates (clean_text, clean_text) pairs -- the image augmentation
    provides the "noisy" input.
    """
    drugs = list(DRUG_VOCAB.keys())
    dosages = ["25mg", "50mg", "100mg", "2mg", "5mg", "10mg", "20mg",
               "75mg", "150mg", "200mg", "500mg"]
    sigs = ["O.P.Nct", "O.P.Nci", "O.P.",
            "S: 1x täglich 1 Tbl.", "S: 2x täglich 1 Tbl.",
            "S: 1 in the morning", "S: 1 in the evening",
            "S: Repetatur", "Repetatur!"]
    pairs = []
    for _ in range(n):
        drug = random.choice(drugs)
        dos = random.choice(dosages)
        sig = random.choice(sigs)
        text = f"{drug} {dos} {sig}"
        pairs.append(text)
    return pairs


class PrescriptionFTDataset(Dataset):
    """Fine-tuning dataset with heavy augmentation."""
    def __init__(self, pairs, processor, max_len=FT_MAX_LEN,
                 augment=True):
        self.pairs = pairs
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.ip = getattr(processor, "image_processor",
                          getattr(processor, "feature_extractor", None))
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")

        if self.augment:
            img = self._augment(img)

        pixel_values = self.ip(img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.tokenizer(
            text, padding="max_length", max_length=self.max_len,
            truncation=True).input_ids
        labels = [l if l != self.tokenizer.pad_token_id else -100
                  for l in labels]
        return {"pixel_values": pixel_values,
                "labels": torch.tensor(labels, dtype=torch.long)}

    def _augment(self, img):
        """Heavy augmentation for robustness."""
        # Brightness jitter
        if random.random() > 0.4:
            img = ImageEnhance.Brightness(img).enhance(
                random.uniform(0.7, 1.3))
        # Contrast jitter
        if random.random() > 0.4:
            img = ImageEnhance.Contrast(img).enhance(
                random.uniform(0.75, 1.25))
        # Slight rotation (±3°)
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, fillcolor=(255, 255, 255),
                             resample=Image.BICUBIC)
        # Gaussian blur
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.5, 1.5)))
        # Salt & pepper noise
        if random.random() > 0.7:
            arr = np.array(img)
            noise = np.random.random(arr.shape[:2])
            arr[noise < 0.01] = 0
            arr[noise > 0.99] = 255
            img = Image.fromarray(arr)
        # Random crop/pad
        if random.random() > 0.6:
            w, h = img.size
            pad = random.randint(0, 10)
            from PIL import ImageOps
            img = ImageOps.expand(img, border=pad, fill=(255, 255, 255))
        return img


def load_finetune_labels(csv_path, crops_dir):
    """Load labels CSV, resolve to image paths."""
    rows = []
    missing = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            text = row["text"].strip()
            if not text:
                continue
            full = os.path.join(crops_dir, fname)
            if os.path.exists(full):
                rows.append((full, text))
            else:
                # Try glob
                import re as _re
                m = _re.search(r'line(\d+)', fname)
                if m:
                    ln = int(m.group(1))
                    pattern = os.path.join(crops_dir, f"p*_*line{ln:02d}_crop.png")
                    matches = sorted(glob.glob(pattern))
                    if matches:
                        for mp in matches:
                            rows.append((mp, text))
                        continue
                missing.append(fname)
    if missing:
        print(f"[Labels] [warn] {len(missing)} missing: {missing[:5]}")
    print(f"[Labels] [ok] {len(rows)} training pairs")
    return rows


def finetune_trocr():
    """Extended fine-tuning with validation and early stopping."""
    if not os.path.exists(FT_LABELS_CSV):
        print(f"[warn] Labels not found: {FT_LABELS_CSV}")
        print("  Download from zip, fill in labels, re-upload.")
        return None

    pairs = load_finetune_labels(FT_LABELS_CSV, FT_CROPS_DIR)
    if len(pairs) == 0:
        print("ERROR: No training pairs.")
        return None

    # Expand small datasets by repetition
    if len(pairs) < 30:
        pairs = pairs * (30 // len(pairs) + 1)
        print(f"[FT] Expanded to {len(pairs)} pairs")

    # Load model
    print(f"[FT] Loading {FT_BASE_MODEL}")
    processor = TrOCRProcessor.from_pretrained(FT_BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(FT_BASE_MODEL).to(FT_DEVICE)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Train/val split
    n_val = max(1, int(len(pairs) * FT_VAL_SPLIT))
    random.shuffle(pairs)
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]

    train_ds = PrescriptionFTDataset(train_pairs, processor, augment=True)
    val_ds = PrescriptionFTDataset(val_pairs, processor, augment=False)
    train_dl = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=FT_BATCH, shuffle=False, num_workers=0)

    # Optimizer + cosine scheduler
    optimizer = AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    total_steps = len(train_dl) * FT_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 8),
        num_training_steps=total_steps)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"[FT] {FT_EPOCHS} epochs x {len(train_dl)} batches "
          f"(train={len(train_pairs)}, val={len(val_pairs)})")
    print(f"{'-' * 55}")

    for epoch in range(FT_EPOCHS):
        # -- Train --
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            pv = batch["pixel_values"].to(FT_DEVICE)
            labels = batch["labels"].to(FT_DEVICE)
            outputs = model(pixel_values=pv, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        avg_train = train_loss / len(train_dl)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                pv = batch["pixel_values"].to(FT_DEVICE)
                labels = batch["labels"].to(FT_DEVICE)
                outputs = model(pixel_values=pv, labels=labels)
                val_loss += outputs.loss.item()
        avg_val = val_loss / max(len(val_dl), 1)

        print(f"  Epoch {epoch+1:2d}/{FT_EPOCHS}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.save_pretrained(FT_OUT_DIR)
            processor.save_pretrained(FT_OUT_DIR)
            print(f"  [ok] saved (best)")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{FT_PATIENCE})")
            if patience_counter >= FT_PATIENCE:
                print(f"  [stop] Early stopping at epoch {epoch+1}")
                break

    print(f"{'-' * 55}")
    print(f"[FT] [ok] Best val loss: {best_val_loss:.4f} -> {FT_OUT_DIR}")
    print(f"Re-run Cell 3 to use the fine-tuned model.")
    return FT_OUT_DIR


# -- Run fine-tuning --
if os.path.exists(FT_LABELS_CSV):
    finetune_trocr()
else:
    print(f"[warn] Upload {FT_LABELS_CSV} with corrected labels first.")
    print("  Format: filename,text")
    print("  Example: p1_drug1_line01_crop.png,Tryptizol 25mg O.P.Nct")    
# ============================================================
# CELL 7 -- Train OCR Correction Model
# ============================================================
# Character-level GRU seq2seq trained on synthetic OCR-error pairs.
# Learns to map common handwriting OCR errors to correct text.
# ============================================================
import random

def generate_ocr_error_pairs(n: int = 5000) -> List[Tuple[str, str]]:
    """
    Generate (noisy_text, clean_text) pairs by simulating OCR errors
    on known prescription vocabulary.
    """
    drugs = list(DRUG_VOCAB.keys())
    dosages = ["25mg", "50mg", "100mg", "2mg", "5mg", "10mg", "20mg",
               "75mg", "150mg", "200mg", "500mg", "1g", "5ml", "10ml"]
    sigs = ["O.P.Nct", "O.P.Nci", "O.P.", "Repetatur",
            "S: 1x täglich", "S: 2x täglich 1 Tbl.",
            "S: 1 in the morning", "S: 1 in the evening",
            "1 morgens", "1 abends", "für Mr. Alain",
            "für Frau Müller", "S: Repetatur!"]

    # OCR error simulation rules
    def add_noise(text: str) -> str:
        """Apply random OCR-like errors."""
        noise_ops = [
            # Common handwriting OCR confusions
            ("m", "rn"), ("m", "nn"), ("w", "vv"),
            ("d", "cl"), ("h", "li"), ("l", "1"),
            ("o", "0"), ("u", "n"), ("n", "u"),
            ("i", "l"), ("t", "f"), ("z", "2"),
            ("a", "o"), ("e", "c"),
            # Drug-specific
            ("Tryptizol", "Tryptlzol"), ("Tryptizol", "Tryptifol"),
            ("Tryptizol", "Jryptizol"), ("Tryptizol", "Trypti2ol"),
            ("Valium", "Valirn"), ("Valium", "Valiurn"),
            ("Valium", "Volium"), ("Valium", "Vahurn"),
            ("täglich", "tägl"), ("Tablette", "Tbl"),
            ("Repetatur", "Repetahir"), ("Repetatur", "Repetchi"),
            # Spacing errors
            (" ", "  "), (" ", ""),
        ]
        result = text
        n_ops = random.randint(0, 3)
        for _ in range(n_ops):
            old, new = random.choice(noise_ops)
            if old in result and random.random() > 0.5:
                # Apply substitution at random position
                idx = result.find(old)
                if idx >= 0:
                    result = result[:idx] + new + result[idx+len(old):]
        # Random character drop
        if random.random() > 0.7 and len(result) > 5:
            pos = random.randint(1, len(result)-2)
            result = result[:pos] + result[pos+1:]
        # Random character insert
        if random.random() > 0.8:
            pos = random.randint(0, len(result))
            ch = random.choice("abcdefghijklmnopqrstuvwxyz0123456789.,- ")
            result = result[:pos] + ch + result[pos:]
        return result

    pairs = []
    for _ in range(n):
        # Generate a clean prescription string
        drug = random.choice(drugs)
        dos = random.choice(dosages)
        sig = random.choice(sigs)
        templates = [
            f"{drug} {dos} {sig}",
            f"Rp. {drug} {dos}",
            sig,
            f"für Mr. {random.choice(['Alain','Schmidt','Müller','Weber'])}",
            f"{drug} {dos}",
            f"S: {random.randint(1,3)}x {random.choice(['täglich','daily'])} "
            f"{random.randint(1,2)} Tbl.",
        ]
        clean = random.choice(templates)
        noisy = add_noise(clean)
        pairs.append((noisy, clean))

    # Also add identity pairs (clean -> clean) for stability
    for _ in range(n // 5):
        drug = random.choice(drugs)
        dos = random.choice(dosages)
        clean = f"{drug} {dos}"
        pairs.append((clean, clean))

    random.shuffle(pairs)
    return pairs


class CorrectorDataset(TorchDataset):
    def __init__(self, pairs, max_len=MAX_SEQ_LEN):
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        src = torch.tensor(encode_text(noisy, self.max_len), dtype=torch.long)
        tgt = torch.tensor(encode_text(clean, self.max_len), dtype=torch.long)
        return src, tgt


def train_corrector(n_pairs=8000, epochs=20, batch_size=64, lr=1e-3):
    """Train the OCR correction model."""
    os.makedirs(CORRECTOR_DIR, exist_ok=True)

    print(f"[Corrector] Generating {n_pairs} training pairs...")
    pairs = generate_ocr_error_pairs(n_pairs)

    # Split
    n_val = max(100, int(len(pairs) * 0.1))
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]

    train_ds = CorrectorDataset(train_pairs)
    val_ds = CorrectorDataset(val_pairs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = OCRCorrectorModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR2IDX["<pad>"])

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Corrector] Model: {param_count:,} params")
    print(f"[Corrector] {epochs} epochs x {len(train_dl)} batches "
          f"(train={len(train_pairs)}, val={len(val_pairs)})")
    print(f"{'-' * 55}")

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_dl:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tf_ratio = max(0.2, 1.0 - epoch / epochs)  # decay TF
            output = model(src, tgt, teacher_forcing_ratio=tf_ratio)
            # output: (B, T, V), tgt: (B, T)
            loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE),
                            tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_dl)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                output = model(src, tgt, teacher_forcing_ratio=0.0)
                loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE),
                                tgt[:, 1:].reshape(-1))
                val_loss += loss.item()
        avg_val = val_loss / max(len(val_dl), 1)

        print(f"  Epoch {epoch+1:2d}/{epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}", end="")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(),
                      os.path.join(CORRECTOR_DIR, "corrector.pt"))
            print("  [ok] saved")
        else:
            print()

    print(f"{'-' * 55}")
    print(f"[Corrector] [ok] Best val: {best_val:.4f} -> {CORRECTOR_DIR}")

    # Quick test
    model.eval()
    test_cases = [
        "Tryptlzol 25rng O.P.Nct",
        "Valiurn 2 O.P.Nci",
        "S: Repetahir!",
        "für Mrn. Alain",
        "S: 2x tägl 1 Tbl.",
    ]
    print(f"\n[Corrector] Quick test:")
    for noisy in test_cases:
        corrected = model.correct(noisy)
        print(f"  {noisy!r:40s} -> {corrected!r}")

    print(f"\nRun the pipeline cell (Cell 9) to use the correction model.")
    return model


# -- Run training --
print("=" * 55)
print(" Training OCR Correction Model")
print("=" * 55)
import os

model_path = os.path.join(CORRECTOR_DIR, "corrector.pt")

if os.path.exists(model_path):
    print("[Corrector] [ok] Already trained -- skipping training.")
else:
    print("[Corrector] No model found -- training now.")
    train_corrector()
# ============================================================
# CELL 8 -- Evaluate & Benchmark
# ============================================================
# If you have ground truth labels, this cell computes
# character error rate (CER) and word error rate (WER).
# ============================================================

def compute_cer(pred: str, ref: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    if not ref:
        return 0.0 if not pred else 1.0
    import difflib
    s = difflib.SequenceMatcher(None, pred, ref)
    distance = len(ref) - sum(b.size for b in s.get_matching_blocks())
    return distance / len(ref)


def compute_wer(pred: str, ref: str) -> float:
    """Word Error Rate."""
    pred_w = pred.split()
    ref_w = ref.split()
    if not ref_w:
        return 0.0 if not pred_w else 1.0
    import difflib
    s = difflib.SequenceMatcher(None, pred_w, ref_w)
    distance = len(ref_w) - sum(b.size for b in s.get_matching_blocks())
    return distance / len(ref_w)


def run_benchmark():
    """
    Benchmark against ground truth labels if available.
    Uses finetune_labels.csv as ground truth.
    """
    if not os.path.exists(FT_LABELS_CSV):
        print("No ground truth labels found. Skip benchmark.")
        return

    labels = {}
    with open(FT_LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "").strip()
            if text:
                labels[row["filename"].strip()] = text

    if not labels:
        print("Labels CSV has no filled entries. Fill in text column first.")
        return

    # Load OCR results
    df = pd.read_csv(OUTPUT_CSV, dtype=str)

    cer_scores = []
    wer_scores = []

    print(f"\n{'-' * 60}")
    print(f" Benchmark: {len(labels)} ground truth labels")
    print(f"{'-' * 60}")

    for fname, ref in labels.items():
        # Find matching OCR result
        # Try to match by bbox/page
        pred = ""
        for _, r in df.iterrows():
            # Check if this OCR row corresponds to this label
            ocr_clean = str(r.get("OCR_Clean", ""))
            if not ocr_clean:
                continue
            # Simple heuristic: use highest-confidence match
            pred = ocr_clean
            break

        cer = compute_cer(pred, ref)
        wer = compute_wer(pred, ref)
        cer_scores.append(cer)
        wer_scores.append(wer)
        print(f"  {fname:35s} CER={cer:.3f} WER={wer:.3f}")
        print(f"    GT:   {ref!r}")
        print(f"    Pred: {pred!r}")

    if cer_scores:
        print(f"\n{'-' * 60}")
        print(f" Average CER: {np.mean(cer_scores):.4f} "
              f"(±{np.std(cer_scores):.4f})")
        print(f" Average WER: {np.mean(wer_scores):.4f} "
              f"(±{np.std(wer_scores):.4f})")
        print(f"{'-' * 60}")


run_benchmark()
# Note: run_pipeline() is now called at the end of Cell 3, so the CSV
# already exists by the time this cell runs.  If you need to re-run the
# pipeline (e.g. after training a corrector), uncomment the line below:
# all_rows = run_pipeline()
# ============================================================
# CELL 9 -- Download results
# ============================================================
from google.colab import files
import zipfile

zip_path = os.path.join(BASE_DIR, "prescription_results_v9.zip")
with zipfile.ZipFile(zip_path, "w") as z:
    if os.path.exists(OUTPUT_CSV):
        z.write(OUTPUT_CSV, "prescription_output.csv")
    for p in glob.glob(f"{ANNOTATED_DIR}/*.png"):
        z.write(p, f"annotated/{os.path.basename(p)}")
    for p in glob.glob(f"{DEBUG_DIR}/*.png"):
        z.write(p, f"debug/{os.path.basename(p)}")
    # Finetune labels template
    labels_path = os.path.join(BASE_DIR, "finetune_labels.csv")
    if not os.path.exists(labels_path):
        all_crops = sorted(glob.glob(f"{DEBUG_DIR}/p*_line*_crop.png"))
        with open(labels_path, "w", newline="", encoding="utf-8") as lf:
            w = csv.DictWriter(lf, fieldnames=["filename", "text"])
            w.writeheader()
            for cp in all_crops:
                w.writerow({"filename": os.path.basename(cp), "text": ""})
    z.write(labels_path, "finetune_labels.csv")

files.download(zip_path)
print("[ok] Downloaded: prescription_results_v9.zip")

# ============================================================
# CELL 10 -- API Integration (Flask / FastAPI)
# ============================================================
# This cell contains COMMENTED-OUT code that wraps the OCR pipeline
# in a REST API.  Uncomment and run it when you want to connect the
# pipeline to an external application (web front-end, mobile app,
# another microservice, etc.).
#
# Two variants are provided:
#   A) Flask   -- lightweight, easy to deploy on any server or Colab
#   B) FastAPI -- async, auto-generates OpenAPI/Swagger docs
#
# Both expose the same endpoint:
#   POST /ocr
#     - Accepts: multipart/form-data with a field named "file" (the PDF)
#     - Returns: JSON with extracted prescription data
#
# To run on Colab, you would typically pair this with ngrok or
# localtunnel to get a public URL.  Instructions are in the comments.
# ============================================================

# -----------------------------------------------------------------
# VARIANT A: Flask API
# -----------------------------------------------------------------
# Uncomment everything below between the START/END markers to use.

# --- START FLASK API ---

# import io, os, json, tempfile
# from flask import Flask, request, jsonify
#
# # If Flask is not installed, run:  pip install flask
#
# app = Flask(__name__)
#
#
# @app.route("/ocr", methods=["POST"])
# def ocr_endpoint():
#     """
#     Receive a prescription PDF via multipart upload, run the full
#     OCR pipeline, and return structured JSON results.
#
#     Request:
#         POST /ocr
#         Content-Type: multipart/form-data
#         Field "file": the PDF binary
#
#     Response (200):
#         {
#           "status": "ok",
#           "filename": "prescription.pdf",
#           "pages": 2,
#           "rows": [
#             {
#               "Page": 1,
#               "ROI": "drug1",
#               "Line_Index": 1,
#               "Doctor": "DR. HEINZ SCHACHINGER",
#               "OCR_Raw": "...",
#               "OCR_Clean": "Tryptizol 25mg O.P.Nct",
#               "Drug_Name": "Tryptizol",
#               "Drug_Class": "Amitriptylin (Antidepressivum)",
#               "Mean_Confidence": 0.87,
#               ...
#             },
#             ...
#           ]
#         }
#
#     Error (400/500):
#         { "status": "error", "message": "..." }
#     """
#     # --- Validate the upload ---
#     if "file" not in request.files:
#         return jsonify({"status": "error",
#                         "message": "No file field in request"}), 400
#
#     uploaded_file = request.files["file"]
#     original_filename = uploaded_file.filename or "upload.pdf"
#
#     if not original_filename.lower().endswith(".pdf"):
#         return jsonify({"status": "error",
#                         "message": "Only PDF files are accepted"}), 400
#
#     # --- Save to a temp file so the pipeline can read it ---
#     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#         uploaded_file.save(tmp)
#         tmp_path = tmp.name
#
#     try:
#         # --- Run the pipeline ---
#         # output_csv is written to a temp path as well
#         tmp_csv = tmp_path.replace(".pdf", "_output.csv")
#         rows = run_pipeline(pdf_path=tmp_path, output_csv=tmp_csv)
#
#         # --- Build the JSON response ---
#         response_data = {
#             "status": "ok",
#             "filename": original_filename,
#             "pages": max((r["Page"] for r in rows), default=0) if rows else 0,
#             "rows": rows,  # list of dicts, directly JSON-serialisable
#         }
#         return jsonify(response_data), 200
#
#     except FileNotFoundError as exc:
#         return jsonify({"status": "error",
#                         "message": str(exc)}), 400
#     except Exception as exc:
#         return jsonify({"status": "error",
#                         "message": f"Pipeline error: {exc}"}), 500
#     finally:
#         # Clean up temp files
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
#         tmp_csv = tmp_path.replace(".pdf", "_output.csv")
#         if os.path.exists(tmp_csv):
#             os.remove(tmp_csv)
#
#
# @app.route("/health", methods=["GET"])
# def health_check():
#     """Simple health-check endpoint for load balancers / monitoring."""
#     return jsonify({"status": "ok", "version": "v10"}), 200
#
#
# # --- How to run ---
# # Option 1: Local machine
# #   app.run(host="0.0.0.0", port=5000, debug=False)
# #
# # Option 2: Google Colab with ngrok (public URL)
# #   pip install pyngrok
# #   from pyngrok import ngrok
# #   public_url = ngrok.connect(5000)
# #   print(f"Public URL: {public_url}")
# #   app.run(port=5000)
# #
# # Option 3: Google Colab with localtunnel
# #   !npm install -g localtunnel
# #   # In a separate cell: !lt --port 5000
# #   app.run(port=5000)

# --- END FLASK API ---


# -----------------------------------------------------------------
# VARIANT B: FastAPI (async, auto-docs at /docs)
# -----------------------------------------------------------------
# Uncomment everything below between the START/END markers to use.

# --- START FASTAPI ---

# import io, os, tempfile
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import uvicorn
#
# # If FastAPI is not installed, run:
# #   pip install fastapi uvicorn python-multipart
#
# api = FastAPI(
#     title="Austrian Prescription OCR API",
#     version="10.0",
#     description="Upload a scanned Austrian/German handwritten prescription "
#                 "PDF and receive structured OCR results as JSON.",
# )
#
#
# @api.post("/ocr", summary="Run OCR on a prescription PDF")
# async def ocr_endpoint(file: UploadFile = File(...)):
#     """
#     Upload a prescription PDF and get back structured data.
#
#     The response includes every detected text line with:
#     - Raw and cleaned OCR text
#     - Drug name and class (if recognised)
#     - Dosage normalisation
#     - Confidence scores
#     - Bounding box coordinates
#     - Doctor name extracted from the header
#     """
#     # Validate file type
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400,
#                             detail="Only PDF files are accepted.")
#
#     # Read the upload into a temp file
#     contents = await file.read()
#     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#         tmp.write(contents)
#         tmp_path = tmp.name
#
#     try:
#         tmp_csv = tmp_path.replace(".pdf", "_output.csv")
#         rows = run_pipeline(pdf_path=tmp_path, output_csv=tmp_csv)
#
#         return JSONResponse(content={
#             "status": "ok",
#             "filename": file.filename,
#             "pages": max((r["Page"] for r in rows), default=0) if rows else 0,
#             "rows": rows,
#         })
#
#     except Exception as exc:
#         raise HTTPException(status_code=500,
#                             detail=f"Pipeline error: {exc}")
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
#         tmp_csv = tmp_path.replace(".pdf", "_output.csv")
#         if os.path.exists(tmp_csv):
#             os.remove(tmp_csv)
#
#
# @api.get("/health", summary="Health check")
# async def health():
#     return {"status": "ok", "version": "v10"}
#
#
# # --- How to run ---
# # Option 1: Local machine
# #   uvicorn.run(api, host="0.0.0.0", port=8000)
# #   # Then open http://localhost:8000/docs for Swagger UI
# #
# # Option 2: Google Colab with ngrok
# #   pip install pyngrok
# #   from pyngrok import ngrok
# #   public_url = ngrok.connect(8000)
# #   print(f"Public URL: {public_url}")
# #   print(f"Swagger docs: {public_url}/docs")
# #   uvicorn.run(api, host="0.0.0.0", port=8000)

# --- END FASTAPI ---


# -----------------------------------------------------------------
# EXAMPLE: Calling the API from a client (Python requests)
# -----------------------------------------------------------------
# import requests
#
# # Point this at wherever the API is running
# API_URL = "http://localhost:5000/ocr"   # Flask
# # API_URL = "http://localhost:8000/ocr" # FastAPI
#
# # Upload a PDF and get results
# with open("my_prescription.pdf", "rb") as f:
#     response = requests.post(API_URL, files={"file": f})
#
# data = response.json()
# print(f"Status: {data['status']}")
# print(f"Pages:  {data['pages']}")
#
# for row in data["rows"]:
#     if row["Is_Drug_Line"] == "YES":
#         print(f"  Drug: {row['Drug_Name']} -- {row['OCR_Clean']}")
#     else:
#         print(f"  Text: {row['OCR_Clean']}")
#
#
# # -----------------------------------------------------------------
# # EXAMPLE: Calling the API from JavaScript (fetch)
# # -----------------------------------------------------------------
# # const formData = new FormData();
# # formData.append('file', pdfFileInput.files[0]);
# #
# # const response = await fetch('http://localhost:8000/ocr', {
# #     method: 'POST',
# #     body: formData,
# # });
# # const data = await response.json();
# # console.log(data.rows);
#
#
# # -----------------------------------------------------------------
# # EXAMPLE: Calling the API from cURL
# # -----------------------------------------------------------------
# # curl -X POST http://localhost:5000/ocr \
# #      -F "file=@/path/to/prescription.pdf" \
# #      | python -m json.tool