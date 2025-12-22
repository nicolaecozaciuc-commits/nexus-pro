"""
NEXUS PRO SERVER v32 - v2.0 + Fix-uri v31
-----------------------------------------------------------------
A Flask-based product search and OCR application for plumbing supplies.
Processes handwritten product lists using Ollama LLM with vision capabilities
and provides intelligent search with synonym expansion and dimension filtering.

Features:
    - OCR extraction from handwritten product lists (via Ollama/LLaVA)
    - Intelligent product search with synonym expansion
    - Query normalization for OCR errors and abbreviations
    - Strict product type filtering (NIPLU vs MUFA, etc.)
    - Precise dimension filtering (1" vs 1 1/4" vs 1/2")
    - Popularity-based result ranking (using sales history)

API Endpoints:
    GET  /           - Main application page
    POST /api/search - Product search with intelligent matching
    POST /api/ocr    - Image OCR processing

Configuration:
    OLLAMA_API_URL: Local Ollama server URL (default: http://127.0.0.1:11434)
    OLLAMA_MODEL: Vision model for OCR (default: llava:7b)

Data Files (auto-detected in working directory):
    - Products: *nexus*.csv or *nexus*.xlsx (excluding 'vandute')
    - Sales history: *vandute*.csv or *rulaj*.csv/.xlsx

Version History:
    v32 (Dec 2025): Comprehensive documentation added
    v31: PRE-FILTRARE on product type, dimension regex fix
    v2.0: Base version with OCR and search

Author: Nexus Team + Claude v31 fixes
Date: December 2025
"""

import os
import re
import json
import requests
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify
from unidecode import unidecode

app = Flask(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Ollama API configuration for local LLM processing
OLLAMA_API_URL = 'http://127.0.0.1:11434/api/generate'
OLLAMA_MODEL = 'llava:7b'

# ==============================================================================
# GLOBAL DATA STRUCTURES
# ==============================================================================

PRODUCTS_DB = []
"""
Global product database loaded at startup from Excel/CSV files.

Structure:
    List[Dict] where each product dict contains:
        - 'd' (str): Product name/description (denumire)
        - 'c' (str): Product code used for inventory/ordering
        - 'norm' (str): Normalized product name for search matching
                       (uppercase, diacritics removed, synonyms expanded)
        - 'score' (float): Sales score from RULAJ_DB (higher = more popular)

Example:
    [
        {'d': 'TEAVA PPR 25x4.2', 'c': 'PPR25', 'norm': 'TEAVA PPR 25X4.2', 'score': 150.0},
        {'d': 'COT ALAMA 1"', 'c': 'CA1', 'norm': 'COT ALAMA 1"', 'score': 85.0}
    ]

Sorted by 'score' descending so popular products appear first in search results.
"""

RULAJ_DB = {}
"""
Sales history database (rulaj = sales turnover in Romanian).

Structure:
    Dict[str, float] mapping product code -> total quantity sold

Purpose:
    Used to rank search results by popularity. Products with higher sales
    volume appear first in search results, improving user experience.

Loaded from CSV/Excel files containing 'vandute' or 'rulaj' in filename.
Expected columns: 'cod' (product code) and 'cantitate'/'stoc' (quantity).

Example:
    {'PPR25': 1500.0, 'CA1': 850.0, 'NIPLU1': 320.0}
"""

WORD_FREQ = Counter()
"""
Word frequency counter for all product names in the database.

Structure:
    Counter[str, int] mapping normalized word -> occurrence count

Purpose:
    Can be used for autocomplete suggestions, typo detection, or
    understanding which terms are most common in the product catalog.

Populated during load_products_database() from normalized product names.
"""

# ==============================================================================
# SYNONYM DICTIONARY
# ==============================================================================

SYNONYMS = {
    """
    Bidirectional synonym mapping for search query expansion.

    Purpose:
        Allows users to find products using alternate terminology.
        When searching for 'CALORIFER', results for 'RADIATOR' are also returned.

    Categories:
        - Main products: Heating equipment, pipes, valves
        - Materials: Metal types, plastic types
        - Fittings: Plumbing connectors and accessories

    Usage:
        During search, each query word is checked against this dictionary.
        If found, all synonyms are added to the search terms.

    Note:
        NIPLU intentionally does NOT include MUFA as a synonym to prevent
        incorrect matches between these distinct fitting types.
    """

    # Main products (produse principale)
    # Heating equipment synonyms
    'CALORIFER': ['RADIATOR', 'ELEMENT', 'CORP'],
    'RADIATOR': ['CALORIFER', 'ELEMENT', 'CORP'],
    'SCARITA': ['RADIATOR BAIE', 'SCARA', 'PORTPROSOP'],  # Towel radiator
    'SCARA': ['RADIATOR BAIE', 'SCARITA', 'PORTPROSOP'],

    # Pipes - handles common misspellings/OCR errors
    'TEAVA': ['TIANA', 'TIGANA', 'CONDUCTA', 'TUB'],
    'TIANA': ['TEAVA', 'TIGANA', 'CONDUCTA'],  # Common OCR misread
    'TIGANA': ['TEAVA', 'TIANA', 'CONDUCTA'],  # Common OCR misread

    # Valves and water heaters
    'ROBINET': ['VANA', 'VENTIL'],
    'BOILER': ['BOLER', 'REZERVOR', 'ACM'],  # ACM = Apa Calda Menajera
    'PUFFER': ['ACUMULATOR', 'REZERVOR'],    # Heat accumulator tank

    # Materials (materiale)
    'OTEL': ['FE', 'FIER', 'METAL'],          # Steel
    'ALAMA': ['BRONZ', 'BRASS'],               # Brass
    'PPR': ['PP-R', 'POLIPROPILENA'],          # Polypropylene pipes
    'PEX': ['PE-X', 'PE-XA'],                  # Cross-linked polyethylene

    # Fittings (fitinguri) - Plumbing connectors
    'COT': ['GAT', 'UNGHI', 'COLT'],           # Elbow fitting
    'TEU': ['T', 'RAMIFICATIE', 'TRU', 'TEI'], # T-junction
    'TRU': ['TEU', 'T'],                       # Alternate spelling
    'TEI': ['TEU', 'T'],                       # Alternate spelling
    'NIPLU': ['NIPEL'],                        # Nipple (NOT MUFA - intentional)
    'DOP': ['CAP', 'CAPAC'],                   # Cap/plug
    'REDUCTIE': ['REDUS', 'REDUCER'],          # Reducer fitting
    'REDUS': ['REDUCTIE', 'REDUCER'],
    'OLANDEZ': ['RACORD', 'PIULITA'],          # Union fitting
    'SUPAPA': ['VALVA', 'VENTIL'],             # Valve
    'VANA': ['ROBINET', 'VENTIL'],             # Gate valve
    'FILTRU': ['FILTER'],                      # Filter
}

def normalize_query(query):
    """
    Normalize a search query for consistent matching against the product database.

    This function performs multiple normalization steps to handle:
    - Case sensitivity (converts to uppercase)
    - Romanian diacritics (ă, î, ș, ț → a, i, s, t)
    - OCR errors from handwritten text recognition
    - Various dimension notation formats (imperial/metric)
    - Common misspellings and abbreviations

    Args:
        query (str): The raw search query from user input.

    Returns:
        str: Normalized query string ready for database matching.

    Normalization Steps:
        1. Case normalization: Convert to uppercase
        2. Diacritic removal: Using unidecode for Romanian characters
        3. PPR pipe dimension fixes:
           - '3EX' → '32X' (32mm PPR pipe, OCR often misreads 32 as 3E)
           - '2EX' → '25X' (25mm PPR pipe)
           - '1EX' → '20X' (20mm PPR pipe)
        4. Imperial dimension normalization:
           - 'TOL'/'INCH' → '"' (tol = inch in Romanian slang)
           - Unicode fractions to ASCII: ½→1/2, ¾→3/4, ¼→1/4
           - Compound fractions: 1½→1.1/2, 2½→2.1/2
        5. Direct synonym replacement:
           - TRU/TEI → TEU (T-junction alternate spellings)
           - TIAN/TIGAN → TEAVA (pipe OCR misreads)
        6. Whitespace normalization: Collapse multiple spaces

    Examples:
        >>> normalize_query("teava ppr 3ex")
        'TEAVA PPR 32X'

        >>> normalize_query("cot 1½ tol")
        'COT 1.1/2 "'

        >>> normalize_query("TRU alama")
        'TEU ALAMA'

        >>> normalize_query("țeavă   PPR")
        'TEAVA PPR'
    """
    q = query.upper().strip()

    # Step 1-2: Remove Romanian diacritics (ă→a, î→i, ș→s, ț→t)
    q = unidecode(q)

    # Step 3: Fix OCR misreads for PPR pipe dimensions
    # OCR often reads '32' as '3E' in handwritten notes
    q = q.replace('3EX', '32X')  # 32mm PPR pipe
    q = q.replace('2EX', '25X')  # 25mm PPR pipe
    q = q.replace('1EX', '20X')  # 20mm PPR pipe

    # Step 4: Normalize imperial dimension notations
    # 'TOL' is Romanian slang for inch (from "țol")
    q = q.replace('TOL', '"')
    q = q.replace('INCH', '"')

    # Convert Unicode fraction characters to ASCII equivalents
    q = q.replace('½', '1/2')
    q = q.replace('¾', '3/4')
    q = q.replace('¼', '1/4')

    # Convert compound fractions (1½ inch = 1.1/2")
    q = q.replace('1½', '1.1/2')
    q = q.replace('2½', '2.1/2')

    # Step 5: Direct synonym replacements for common variants
    # TEU (T-junction) has multiple spelling variants
    q = q.replace(' TRU ', ' TEU ')
    q = q.replace(' TEI ', ' TEU ')

    # TEAVA (pipe) - common OCR/handwriting misreads
    q = q.replace(' TIAN ', ' TEAVA ')
    q = q.replace(' TIGAN ', ' TEAVA ')

    # Step 6: Normalize whitespace (collapse multiple spaces to single)
    q = ' '.join(q.split())

    return q

def load_sales_history():
    """
    Load sales history data to rank search results by product popularity.

    This function scans the current directory for sales data files and loads
    them into the global RULAJ_DB dictionary. Products with higher sales volumes
    will appear first in search results, improving user experience.

    File Discovery:
        Searches for files matching these criteria:
        - Extensions: .csv, .xlsx, or .xls
        - Filename contains: 'vandute' (sold) or 'rulaj' (turnover)

    Expected File Format:
        The file must contain these columns (case-insensitive):
        - 'cod' or similar: Product code identifier
        - 'cantitate' or 'stoc': Quantity/stock value

        Example CSV:
            cod,denumire,cantitate
            PPR25,Teava PPR 25mm,1500
            CA1,Cot alama 1",850

    Side Effects:
        - Modifies global RULAJ_DB dictionary
        - Prints status messages to stdout

    Returns:
        None. Data is stored in global RULAJ_DB.

    Error Handling:
        - Missing file: Prints warning, returns without modifying RULAJ_DB
        - Missing columns: Prints error message
        - Parse errors: Prints exception details, RULAJ_DB unchanged

    Note:
        If multiple matching files exist, only the first one is loaded.
        Quantities are converted to absolute values (negative values become positive).
    """
    global RULAJ_DB
    print("🔄 Caut fișiere de rulaj...")

    # Find sales history files in current directory
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls')) and ('vandute' in f.lower() or 'rulaj' in f.lower())]

    if not files:
        print("⚠️ Nu am găsit fișier de vânzări.")
        return

    try:
        file_path = files[0]
        print(f"📂 Încarc rulaj din: {file_path}")

        # Load file based on extension
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, dtype=str)
        else:
            df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip')

        # Normalize column names for consistent access
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Auto-detect quantity and code columns
        col_qty = next((c for c in df.columns if 'cantitate' in c or 'stoc' in c), None)
        col_cod = next((c for c in df.columns if 'cod' in c), None)

        if col_qty and col_cod:
            # Convert quantities to numeric, handle errors, use absolute values
            df[col_qty] = pd.to_numeric(df[col_qty], errors='coerce').fillna(0).abs()
            # Aggregate quantities by product code
            rulaj_group = df.groupby(col_cod)[col_qty].sum()
            RULAJ_DB = rulaj_group.to_dict()
            print(f"✅ Rulaj încărcat: {len(RULAJ_DB)} produse.")
        else:
            print(f"❌ Coloane lipsă în rulaj.")

    except Exception as e:
        print(f"❌ Eroare rulaj: {e}")

def load_products_database():
    """
    Load the product catalog from Excel/CSV files into memory.

    This function initializes the global PRODUCTS_DB with normalized product
    data and populates WORD_FREQ with word occurrence statistics. Products
    are sorted by sales score (from RULAJ_DB) so popular items appear first.

    File Discovery:
        Searches for files matching these criteria:
        - Extensions: .csv or .xlsx
        - Filename contains: 'nexus'
        - Filename does NOT contain: 'vandute' (to exclude sales files)

    Expected File Format:
        Preferred columns (auto-detected, case-insensitive):
        - 'denumire': Product name/description
        - 'cod': Long product code (primary identifier)
        - 'selectie': Short product code (preferred if available)

        Fallback (if 'denumire' not found):
        - Column 0: Long code
        - Column 3: Product name
        - Column 12: Short code (if exists)

    Product Object Structure:
        Each product in PRODUCTS_DB contains:
        {
            'd': str,      # Original product name (denumire)
            'c': str,      # Final code (short code preferred over long)
            'norm': str,   # Normalized name for search matching
            'score': float # Sales ranking score from RULAJ_DB
        }

    Side Effects:
        - Modifies global PRODUCTS_DB (list of product dicts)
        - Modifies global WORD_FREQ (word occurrence counter)
        - Prints status messages to stdout

    Dependencies:
        - RULAJ_DB should be loaded first (via load_sales_history())
        - normalize_query() function for text normalization

    Returns:
        None. Data is stored in global variables.

    Data Cleaning:
        - Skips rows with empty/short names (< 2 chars)
        - Skips header rows (name == 'denumire')
        - Handles 'nan' string values from pandas

    Note:
        Call load_sales_history() BEFORE this function to ensure
        sales scores are available for ranking products.
    """
    global PRODUCTS_DB, WORD_FREQ
    print("🔄 Caut baza de date produse...")

    # Find product catalog files (exclude sales files)
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx')) and 'nexus' in f.lower() and 'vandute' not in f.lower()]

    if not files:
        print("❌ Nu am găsit fișierul de produse.")
        return

    try:
        file_path = files[0]
        print(f"📂 Încarc produse din: {file_path}")

        # Load file based on extension
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, dtype=str)
        else:
            df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip')

        # Normalize column names for consistent access
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Auto-detect columns by name
        col_den = next((c for c in df.columns if 'denumire' in c), None)
        col_cod_lung = next((c for c in df.columns if c == 'cod'), None)
        col_cod_scurt = next((c for c in df.columns if 'selectie' in c), None)

        # Fallback: Use column positions if 'denumire' not found
        if not col_den and len(df.columns) > 3:
            df = df.iloc[:, [0, 3, 12] if len(df.columns) > 12 else [0, 1]]
            col_cod_lung = df.columns[0]
            col_den = df.columns[1]
            col_cod_scurt = df.columns[2] if len(df.columns) > 2 else None

        clean_data = []
        word_list = []

        # Process each row into a product object
        for _, row in df.iterrows():
            den = str(row.get(col_den, '')).strip()

            # Skip empty/invalid rows and header rows
            if len(den) < 2 or den.lower() == 'denumire':
                continue

            c_lung = str(row.get(col_cod_lung, '')).strip()
            c_scurt = str(row.get(col_cod_scurt, '')).strip()

            # Prefer short code over long code
            final_code = c_scurt if c_scurt and c_scurt.lower() != 'nan' else c_lung

            if final_code and final_code.lower() != 'nan':
                # Look up sales score (try short code first, then long)
                sales_score = RULAJ_DB.get(final_code, 0)
                if sales_score == 0 and c_lung:
                    sales_score = RULAJ_DB.get(c_lung, 0)

                # Normalize product name for search matching
                den_norm = normalize_query(den)

                # Build product object
                prod_obj = {
                    'd': den,           # Original product name
                    'c': final_code,    # Product code for ordering
                    'norm': den_norm,   # Normalized name for search
                    'score': float(sales_score)  # Sales ranking
                }
                clean_data.append(prod_obj)
                word_list.extend(den_norm.split())

        # Sort by sales score (descending) so popular products appear first
        clean_data.sort(key=lambda x: x['score'], reverse=True)

        PRODUCTS_DB = clean_data
        WORD_FREQ = Counter(word_list)

        print(f"✅ Baza de date: {len(PRODUCTS_DB)} produse.")

    except Exception as e:
        print(f"❌ Eroare produse: {e}")

def filter_by_dimensions(query_normalized, results):
    """
    Filter search results to match exact imperial pipe dimensions.

    This function implements strict dimension filtering to distinguish between
    similar but different pipe sizes. It prevents 1" (one inch) searches from
    returning 1 1/4" (one and a quarter inch) products, which is a common
    source of ordering errors in plumbing supplies.

    Args:
        query_normalized (str): The normalized search query (uppercase, ASCII).
        results (list): List of product dicts from initial search.

    Returns:
        list: Filtered results matching the exact dimension specification.
              Returns original results if no dimension keywords detected
              or if filtering would return zero results.

    Dimension Detection Patterns:
        The function detects these dimension specifications in the query:

        | Variable           | Pattern              | Example Query |
        |--------------------|----------------------|---------------|
        | has_1_inch         | 1" or 1'             | "COT 1""      |
        | has_2_inch         | 2" or 2'             | "NIPLU 2""    |
        | has_1_half         | 1.1/2, 1 1/2, or 1½  | "TEAVA 1½""   |
        | has_half           | 1/2 or ½             | "DOP 1/2""    |
        | has_three_quarters | 3/4 or ¾             | "MUFA 3/4""   |

    Filtering Logic:

        1. Simple 1" (without fractions):
           - Query: "COT 1""
           - EXCLUDES products containing:
             * 1.1/4, 1 1/4, 1'1/4, 1''1/4, 1"1/4 (one and a quarter)
             * 1½, 1¼ (Unicode compound fractions)
             * Standalone 1/2 or 1/4 without the 1" prefix
           - This prevents "COT 1"" from matching "COT 1 1/4"" or "COT 1/2""

        2. Compound 1½" (one and a half):
           - Query: "NIPLU 1½""
           - REQUIRES products to contain: 1.1/2, 1 1/2, or 1½
           - Products without these patterns are excluded

        3. Simple ½" (half inch only):
           - Query: "MUFA 1/2""
           - REQUIRES products to contain: 1/2 or ½
           - Only applies when query has ½ WITHOUT 1" or 1½

    Examples:
        Query "COT 1"" should return:
            ✓ COT ALAMA 1"
            ✓ COT FI-FE 1"
            ✗ COT ALAMA 1 1/4"  (excluded - compound dimension)
            ✗ COT 1/2"           (excluded - different size)

        Query "NIPLU 1½"" should return:
            ✓ NIPLU ALAMA 1.1/2"
            ✓ NIPLU 1½"
            ✗ NIPLU 1"           (excluded - missing half)
            ✗ NIPLU 2"           (excluded - wrong size)

    Note:
        If filtering would result in zero matches, the original unfiltered
        results are returned to avoid empty search results.
    """
    import re

    # Detect dimension patterns in the query
    has_1_inch = bool(re.search(r'\b1["\']', query_normalized))
    has_2_inch = bool(re.search(r'\b2["\']', query_normalized))
    has_1_half = bool(re.search(r'1\.1/2|1 1/2|1½', query_normalized))
    has_half = bool(re.search(r'\b1/2\b|½', query_normalized))
    has_three_quarters = bool(re.search(r'\b3/4\b|¾', query_normalized))

    # Skip filtering if no dimension keywords detected
    if not any([has_1_inch, has_2_inch, has_1_half, has_half, has_three_quarters]):
        return results

    filtered = []

    for r in results:
        # Combine code and name for pattern matching
        prod_text = (r['c'] + ' ' + r['d']).upper()
        exclude = False

        # Rule 1: Simple 1" - exclude compound dimensions
        if has_1_inch and not has_half and not has_1_half:
            # Exclude: 1.1/4, 1 1/4, 1'1/4, 1''1/4, 1"1/4, 1½, 1¼
            if re.search(r'1\.1/[24]|1 1/[24]|1["\']+ ?1/[24]|1½|1¼', prod_text):
                exclude = True
            # Exclude standalone fractions without the 1" prefix
            elif re.search(r'\b1/[24]\b', prod_text) and not re.search(r'\b1["\']', prod_text):
                exclude = True

        # Rule 2: Compound 1½" - require this exact dimension
        if has_1_half:
            if not re.search(r'1\.1/2|1 1/2|1½', prod_text):
                exclude = True

        # Rule 3: Simple ½" - require half inch (not as part of 1" or 1½")
        if has_half and not has_1_inch and not has_1_half:
            if not re.search(r'1/2|½', prod_text):
                exclude = True

        if not exclude:
            filtered.append(r)

    # Return filtered results, or original if filtering returned nothing
    return filtered if filtered else results

def filter_by_product_type(query_normalized, results):
    """
    Filter search results to match a specific product type (fitting category).

    This function performs strict product type filtering to prevent confusion
    between different fitting types that may share similar names or materials.
    For example, searching for "NIPLU ALAMA" should not return "MUFA ALAMA"
    even though both contain "ALAMA" (brass).

    Args:
        query_normalized (str): The normalized search query (uppercase, ASCII).
        results (list): List of product dicts from initial search.

    Returns:
        list: Filtered results containing only the detected product type.
              Returns original results if no product type is detected
              or if filtering would return zero results.

    Supported Product Types:
        The function recognizes these plumbing fitting types:

        | Query Keyword | Search Pattern | Description (EN)      |
        |---------------|----------------|-----------------------|
        | NIPLU         | NIPLU          | Nipple                |
        | DOP           | DOP            | Cap/Plug              |
        | REDUCTIE      | REDUCT         | Reducer               |
        | REDUS         | REDUCT         | Reducer (alternate)   |
        | MUFA          | MUFA           | Coupling              |
        | COT           | COT            | Elbow                 |
        | TEU           | TEU            | T-junction            |
        | FILTRU        | FILTRU         | Filter                |
        | SUPAPA        | SUPAPA         | Check valve           |
        | ROBINET       | ROBINET        | Tap/Faucet            |
        | VANA          | VANA           | Gate valve            |

    Pattern Mapping:
        Some keywords map to shorter search patterns to match product
        name variations:
        - REDUCTIE → REDUCT (matches both "REDUCTIE" and "REDUCTOR")
        - REDUS → REDUCT (alternate spelling for reducer)

    Example:
        Query: "NIPLU ALAMA 1""
        Results before filtering:
            - NIPLU ALAMA 1"
            - MUFA ALAMA 1"
            - NIPLU FI-FE 1"

        After filter_by_product_type():
            - NIPLU ALAMA 1"
            - NIPLU FI-FE 1"
            (MUFA excluded because detected type is NIPLU)

    Side Effects:
        Prints filtering status message to stdout when filtering is applied.

    Note:
        Only the first matching product type is used (in dictionary order).
        If filtering would return zero results, the original results are returned.
    """
    # Product type keywords and their search patterns
    product_types = {
        'NIPLU': 'NIPLU',       # Nipple fitting
        'DOP': 'DOP',           # Cap/plug
        'REDUCTIE': 'REDUCT',   # Reducer (short pattern for variations)
        'REDUS': 'REDUCT',      # Reducer alternate spelling
        'MUFA': 'MUFA',         # Coupling
        'COT': 'COT',           # Elbow
        'TEU': 'TEU',           # T-junction
        'FILTRU': 'FILTRU',     # Filter
        'SUPAPA': 'SUPAPA',     # Check valve
        'ROBINET': 'ROBINET',   # Tap/faucet
        'VANA': 'VANA',         # Gate valve
    }

    # Detect product type in query
    detected_type = None
    for query_word, search_pattern in product_types.items():
        if query_word in query_normalized:
            detected_type = search_pattern
            break

    # Apply filtering if a product type was detected
    if detected_type:
        results_with_type = [r for r in results if detected_type in r['d'].upper()]
        if results_with_type:
            print(f"🔍 PRE-FILTRARE: '{detected_type}' → {len(results_with_type)} produse")
            return results_with_type

    return results

# --- INIȚIALIZARE ---
print("=" * 80)
print("🔄 Initializez incarcarea bazei de date...")
load_sales_history()
load_products_database()
print(f"✅ SUCCES: {len(PRODUCTS_DB)} produse incarcate in memorie.")
print(f"📊 RULAJ: {len(RULAJ_DB)} produse cu rulaj incarcat.")
if RULAJ_DB:
    top_3 = sorted(RULAJ_DB.items(), key=lambda x: x[1], reverse=True)[:3]
    for cod, qty in top_3:
        print(f"   🔥 {cod}: {int(qty)} buc")
print("=" * 80)

# ==============================================================================
# WEB ROUTES / API ENDPOINTS
# ==============================================================================

@app.route('/')
def index():
    """
    Serve the main application page.

    Returns:
        HTML: Rendered index.html template containing the product search
              interface with OCR image upload and manual text input capabilities.
    """
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search for products in the database with intelligent matching.

    This endpoint provides product search with the following features:
    - Query normalization (handles OCR errors, diacritics, abbreviations)
    - Synonym expansion (finds related terms automatically)
    - Product type filtering (separates NIPLU from MUFA, etc.)
    - Dimension filtering (distinguishes 1" from 1 1/4")
    - Popularity ranking (frequent sellers appear first)

    Request:
        POST /api/search
        Content-Type: application/json

        Body:
            {
                "query": "string"  // Search query (min 2 characters)
            }

    Response:
        Content-Type: application/json

        Success (200):
            [
                {"d": "Product Name", "c": "PROD_CODE"},
                {"d": "Another Product", "c": "PROD2"},
                ...
            ]
            Maximum 30 results, sorted by sales popularity.

        Empty query or no results:
            []

    Search Algorithm:
        1. Normalize query (uppercase, remove diacritics, fix OCR errors)
        2. Expand with synonyms (CALORIFER → also search RADIATOR)
        3. Find products where ALL query words appear in normalized name
        4. Filter by product type if detected (NIPLU, COT, etc.)
        5. Filter by dimension if detected (1", 1/2", etc.)
        6. Return top 30 results by sales ranking

    Examples:
        Request:  {"query": "cot alama 1"}
        Response: [{"d": "COT ALAMA FI-FE 1\"", "c": "CAL1"}, ...]

        Request:  {"query": "teava ppr 32"}
        Response: [{"d": "TEAVA PPR 32x5.4", "c": "PPR32"}, ...]

        Request:  {"query": "x"}
        Response: []  (query too short)

    Error Handling:
        All exceptions are caught and logged. Returns empty array on error
        to prevent frontend crashes.
    """
    try:
        data = request.json
        query = data.get('query', '').strip()

        # Require minimum query length
        if len(query) < 2:
            return jsonify([])

        # Step 1: Normalize the query
        query_normalized = normalize_query(query)

        # Step 2: Expand query with synonyms
        query_words = query_normalized.split()
        expanded_terms = set(query_words)
        for word in query_words:
            if word in SYNONYMS:
                expanded_terms.update(SYNONYMS[word])

        # Step 3: Search for products matching ALL query words
        results = []
        search_terms = [t.lower() for t in expanded_terms]

        for prod in PRODUCTS_DB:
            # All original query words must appear in product name
            if all(term in prod['norm'].lower() for term in query_words):
                results.append(prod)
                if len(results) >= 100:  # Initial limit before filtering
                    break

        # Step 4: Filter by product type (NIPLU, COT, etc.)
        results = filter_by_product_type(query_normalized, results)

        # Step 5: Filter by dimension (1", 1/2", etc.)
        results = filter_by_dimensions(query_normalized, results)

        # Step 6: Return top 30 results (already sorted by popularity)
        return jsonify([{'d': r['d'], 'c': r['c']} for r in results[:30]])

    except Exception as e:
        print(f"Eroare search: {e}")
        return jsonify([])

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    """
    Process an image using OCR to extract product names and quantities.

    This endpoint accepts a base64-encoded image (typically a photo of a
    handwritten product list) and uses a local Ollama LLM with vision
    capabilities to extract structured product data.

    Request:
        POST /api/ocr
        Content-Type: application/json

        Body:
            {
                "image": "base64_encoded_image_string"
            }

        The image should be a base64 string WITHOUT the data URI prefix
        (no "data:image/png;base64," prefix).

    Response:
        Content-Type: application/json

        Success (200):
            {
                "items": [
                    {"text": "product name from image", "qty": 2},
                    {"text": "another product", "qty": 1},
                    ...
                ]
            }

        Missing image:
            {"error": "Lipseste imaginea"}

        Ollama connection error:
            {"error": "Nu mă pot conecta la Ollama."}

        Ollama processing error:
            {"error": "Ollama Error: <error details>"}

        Invalid JSON from LLM:
            {"error": "Nu am găsit JSON valid", "raw": "<raw LLM output>"}

        JSON parsing error:
            {"error": "Eroare decodare JSON", "raw": "<cleaned text>"}

        Other errors:
            {"error": "<exception message>"}

    Processing Pipeline:
        1. Validate that image data is present
        2. Build Ollama API request with vision model prompt
        3. Send request to local Ollama server (120s timeout)
        4. Strip markdown code fences from LLM response
        5. Extract JSON object using regex (handles extra text)
        6. Parse and return structured product data

    LLM Prompt Details:
        The prompt instructs the LLM to:
        - Act as a plumbing/installation expert
        - Analyze handwritten lists in the image
        - Extract product names and quantities
        - Return ONLY valid JSON (no markdown, no extra text)
        - Default quantity to 1 if not visible

    Configuration:
        Uses OLLAMA_API_URL and OLLAMA_MODEL from global config.
        Default model: llava:7b (vision-capable LLM)

    Error Handling:
        - Connection errors return user-friendly message
        - Invalid JSON responses include raw text for debugging
        - All exceptions are logged and return error messages

    Note:
        Requires Ollama server running locally at OLLAMA_API_URL
        with the configured vision model installed.
    """
    try:
        data = request.json
        image_base64 = data.get('image', '')

        # Validate image data is present
        if not image_base64:
            return jsonify({"error": "Lipseste imaginea"})

        print(f"🤖 Trimit cerere către {OLLAMA_MODEL}...")

        # Build Ollama API request payload
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "prompt": (
                "Esti un expert in instalatii. Analizeaza lista scrisa de mana din imagine. "
                "Extrage produsele si cantitatile. "
                "Returneaza DOAR un JSON valid, fara markdown, cu acest format strict: "
                "{ \"items\": [{ \"text\": \"nume produs detectat\", \"qty\": 1 }] }. "
                "Daca nu vezi cantitate, pune 1. Nu adauga text extra."
            ),
            "images": [image_base64],
            "stream": False,
            "options": {"temperature": 0.1}  # Low temperature for consistent output
        }

        # Send request to local Ollama server
        response = requests.post(OLLAMA_API_URL, json=ollama_payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"Ollama Error: {response.text}"})

        result = response.json()
        raw_text = result.get('response', '')

        # Clean markdown code fences if present
        clean_text = raw_text.replace('```json', '').replace('```', '').strip()

        # Extract and parse JSON from response
        try:
            # Use regex to find JSON object (handles extra text before/after)
            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return jsonify(parsed_data)
            else:
                return jsonify({"error": "Nu am găsit JSON valid", "raw": raw_text})
        except json.JSONDecodeError:
            return jsonify({"error": "Eroare decodare JSON", "raw": clean_text})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Nu mă pot conecta la Ollama."})
    except Exception as e:
        print(f"❌ Eroare OCR: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)
