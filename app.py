"""
NEXUS PRO SERVER v32 - v2.0 + Fix-uri v31
-----------------------------------------------------------------
Îmbunătățiri față de v2.0:
1. PRE-FILTRARE strictă pe TIP produs (NIPLU exclude MUFA)
2. FILTRARE dimensiuni cu regex complet (1''1/4, 1"1/4, etc)
3. Sinonime corecte + normalizări (TEU→TEI, 3ex→32x, etc)
4. Reguli prioritare (TERMO+, 22K, VR, WILO)

Autor: Nexus Team + Claude v31 fixes
Data: Decembrie 2025
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

# --- CONFIGURAȚIE ---
OLLAMA_API_URL = 'http://127.0.0.1:11434/api/generate'
OLLAMA_MODEL = 'llava:7b'

# Structuri de date globale
PRODUCTS_DB = []
RULAJ_DB = {}
WORD_FREQ = Counter()

# --- SINONIME CORECTE (din v31) ---
SYNONYMS = {
    # Produse principale
    'CALORIFER': ['RADIATOR', 'ELEMENT', 'CORP'],
    'RADIATOR': ['CALORIFER', 'ELEMENT', 'CORP'],
    'SCARITA': ['RADIATOR BAIE', 'SCARA', 'PORTPROSOP'],
    'SCARA': ['RADIATOR BAIE', 'SCARITA', 'PORTPROSOP'],
    'TEAVA': ['TIANA', 'TIGANA', 'CONDUCTA', 'TUB'],
    'TIANA': ['TEAVA', 'TIGANA', 'CONDUCTA'],
    'TIGANA': ['TEAVA', 'TIANA', 'CONDUCTA'],
    'ROBINET': ['VANA', 'VENTIL'],
    'BOILER': ['BOLER', 'REZERVOR', 'ACM'],
    'PUFFER': ['ACUMULATOR', 'REZERVOR'],
    
    # Materiale
    'OTEL': ['FE', 'FIER', 'METAL'],
    'ALAMA': ['BRONZ', 'BRASS'],
    'PPR': ['PP-R', 'POLIPROPILENA'],
    'PEX': ['PE-X', 'PE-XA'],
    
    # Fitinguri - CORECTATE
    'COT': ['GAT', 'UNGHI', 'COLT'],
    'TEU': ['T', 'RAMIFICATIE', 'TRU', 'TEI'],
    'TRU': ['TEU', 'T'],
    'TEI': ['TEU', 'T'],
    'NIPLU': ['NIPEL'],  # NU include MUFA!
    'DOP': ['CAP', 'CAPAC'],
    'REDUCTIE': ['REDUS', 'REDUCER'],
    'REDUS': ['REDUCTIE', 'REDUCER'],
    'OLANDEZ': ['RACORD', 'PIULITA'],
    'SUPAPA': ['VALVA', 'VENTIL'],
    'VANA': ['ROBINET', 'VENTIL'],
    'FILTRU': ['FILTER'],
}

def normalize_query(query):
    """Normalizare avansată query (din v31)"""
    q = query.upper().strip()
    
    # Normalizări diacritice
    q = unidecode(q)
    
    # Normalizări dimensiuni PPR
    q = q.replace('3EX', '32X')
    q = q.replace('2EX', '25X')
    q = q.replace('1EX', '20X')
    
    # Normalizări inch/fracții
    q = q.replace('TOL', '"')
    q = q.replace('INCH', '"')
    q = q.replace('½', '1/2')
    q = q.replace('¾', '3/4')
    q = q.replace('¼', '1/4')
    q = q.replace('1½', '1.1/2')
    q = q.replace('2½', '2.1/2')
    
    # Normalizări sinonime directe
    q = q.replace(' TRU ', ' TEU ')
    q = q.replace(' TEI ', ' TEU ')
    q = q.replace(' TIAN ', ' TEAVA ')
    q = q.replace(' TIGAN ', ' TEAVA ')
    
    # Spații multiple
    q = ' '.join(q.split())
    
    return q

def load_sales_history():
    """Încarcă rulaj (identic cu v2.0)"""
    global RULAJ_DB
    print("🔄 Caut fișiere de rulaj...")
    
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls')) and ('vandute' in f.lower() or 'rulaj' in f.lower())]
    
    if not files:
        print("⚠️ Nu am găsit fișier de vânzări.")
        return

    try:
        file_path = files[0]
        print(f"📂 Încarc rulaj din: {file_path}")
        
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, dtype=str)
        else:
            df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip')

        df.columns = [str(c).strip().lower() for c in df.columns]
        
        col_qty = next((c for c in df.columns if 'cantitate' in c or 'stoc' in c), None)
        col_cod = next((c for c in df.columns if 'cod' in c), None)

        if col_qty and col_cod:
            df[col_qty] = pd.to_numeric(df[col_qty], errors='coerce').fillna(0).abs()
            rulaj_group = df.groupby(col_cod)[col_qty].sum()
            RULAJ_DB = rulaj_group.to_dict()
            print(f"✅ Rulaj încărcat: {len(RULAJ_DB)} produse.")
        else:
            print(f"❌ Coloane lipsă în rulaj.")

    except Exception as e:
        print(f"❌ Eroare rulaj: {e}")

def load_products_database():
    """Încarcă produse (îmbunătățit cu normalizare)"""
    global PRODUCTS_DB, WORD_FREQ
    print("🔄 Caut baza de date produse...")
    
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx')) and 'nexus' in f.lower() and 'vandute' not in f.lower()]
    
    if not files:
        print("❌ Nu am găsit fișierul de produse.")
        return

    try:
        file_path = files[0]
        print(f"📂 Încarc produse din: {file_path}")
        
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, dtype=str)
        else:
            df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip')

        df.columns = [str(c).strip().lower() for c in df.columns]
        
        col_den = next((c for c in df.columns if 'denumire' in c), None)
        col_cod_lung = next((c for c in df.columns if c == 'cod'), None)
        col_cod_scurt = next((c for c in df.columns if 'selectie' in c), None)
        
        if not col_den and len(df.columns) > 3:
            df = df.iloc[:, [0, 3, 12] if len(df.columns) > 12 else [0, 1]]
            col_cod_lung = df.columns[0]
            col_den = df.columns[1]
            col_cod_scurt = df.columns[2] if len(df.columns) > 2 else None

        clean_data = []
        word_list = []

        for _, row in df.iterrows():
            den = str(row.get(col_den, '')).strip()
            if len(den) < 2 or den.lower() == 'denumire': 
                continue
            
            c_lung = str(row.get(col_cod_lung, '')).strip()
            c_scurt = str(row.get(col_cod_scurt, '')).strip()
            
            final_code = c_scurt if c_scurt and c_scurt.lower() != 'nan' else c_lung
            
            if final_code and final_code.lower() != 'nan':
                sales_score = RULAJ_DB.get(final_code, 0)
                if sales_score == 0 and c_lung:
                    sales_score = RULAJ_DB.get(c_lung, 0)

                # Normalizare denumire pentru căutare
                den_norm = normalize_query(den)
                
                prod_obj = {
                    'd': den,
                    'c': final_code,
                    'norm': den_norm,  # Pentru căutare
                    'score': float(sales_score)
                }
                clean_data.append(prod_obj)
                word_list.extend(den_norm.split())

        clean_data.sort(key=lambda x: x['score'], reverse=True)
        
        PRODUCTS_DB = clean_data
        WORD_FREQ = Counter(word_list)
        
        print(f"✅ Baza de date: {len(PRODUCTS_DB)} produse.")
        
    except Exception as e:
        print(f"❌ Eroare produse: {e}")

def filter_by_dimensions(query_normalized, results):
    """
    FIX v31: FILTRARE STRICTĂ dimensiuni SIMPLE vs COMPUSE
    """
    import re
    
    # Detectare dimensiuni SIMPLE
    has_1_inch = bool(re.search(r'\b1["\']', query_normalized))
    has_2_inch = bool(re.search(r'\b2["\']', query_normalized))
    has_1_half = bool(re.search(r'1\.1/2|1 1/2|1½', query_normalized))
    has_half = bool(re.search(r'\b1/2\b|½', query_normalized))
    has_three_quarters = bool(re.search(r'\b3/4\b|¾', query_normalized))
    
    if not any([has_1_inch, has_2_inch, has_1_half, has_half, has_three_quarters]):
        return results
    
    filtered = []
    
    for r in results:
        prod_text = (r['c'] + ' ' + r['d']).upper()
        exclude = False
        
        # Când query are 1" SIMPLU
        if has_1_inch and not has_half and not has_1_half:
            # Regex COMPLET (v31): exclude 1.1/4, 1 1/4, 1'1/4, 1''1/4, 1"1/4
            if re.search(r'1\.1/[24]|1 1/[24]|1["\']+ ?1/[24]|1½|1¼', prod_text):
                exclude = True
            elif re.search(r'\b1/[24]\b', prod_text) and not re.search(r'\b1["\']', prod_text):
                exclude = True
        
        # Când query are 1½ EXPLICIT
        if has_1_half:
            if not re.search(r'1\.1/2|1 1/2|1½', prod_text):
                exclude = True
        
        # Când query are ½ SIMPLU (fără 1)
        if has_half and not has_1_inch and not has_1_half:
            if not re.search(r'1/2|½', prod_text):
                exclude = True
        
        if not exclude:
            filtered.append(r)
    
    return filtered if filtered else results

def filter_by_product_type(query_normalized, results):
    """
    FIX v31: PRE-FILTRARE pe TIP produs
    """
    product_types = {
        'NIPLU': 'NIPLU',
        'DOP': 'DOP',
        'REDUCTIE': 'REDUCT',
        'REDUS': 'REDUCT',
        'MUFA': 'MUFA',
        'COT': 'COT',
        'TEU': 'TEU',
        'FILTRU': 'FILTRU',
        'SUPAPA': 'SUPAPA',
        'ROBINET': 'ROBINET',
        'VANA': 'VANA',
    }
    
    detected_type = None
    for query_word, search_pattern in product_types.items():
        if query_word in query_normalized:
            detected_type = search_pattern
            break
    
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

# --- RUTE WEB ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """
    API Search cu fix-uri v31
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if len(query) < 2:
            return jsonify([])
        
        # Normalizare query
        query_normalized = normalize_query(query)
        
        # Expandare cu sinonime
        query_words = query_normalized.split()
        expanded_terms = set(query_words)
        for word in query_words:
            if word in SYNONYMS:
                expanded_terms.update(SYNONYMS[word])
        
        # Căutare
        results = []
        search_terms = [t.lower() for t in expanded_terms]
        
        for prod in PRODUCTS_DB:
            # Match toate cuvintele
            if all(term in prod['norm'].lower() for term in query_words):
                results.append(prod)
                if len(results) >= 100:  # Limită inițială
                    break
        
        # FIX v31: PRE-FILTRARE pe TIP
        results = filter_by_product_type(query_normalized, results)
        
        # FIX v31: FILTRARE dimensiuni
        results = filter_by_dimensions(query_normalized, results)
        
        # Returnează top 30
        return jsonify([{'d': r['d'], 'c': r['c']} for r in results[:30]])
        
    except Exception as e:
        print(f"Eroare search: {e}")
        return jsonify([])

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    """OCR processing (identic cu v2.0)"""
    try:
        data = request.json
        image_base64 = data.get('image', '')
        
        if not image_base64:
            return jsonify({"error": "Lipseste imaginea"})

        print(f"🤖 Trimit cerere către {OLLAMA_MODEL}...")
        
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
            "options": {"temperature": 0.1}
        }
        
        response = requests.post(OLLAMA_API_URL, json=ollama_payload, timeout=120)
        
        if response.status_code != 200:
            return jsonify({"error": f"Ollama Error: {response.text}"})
            
        result = response.json()
        raw_text = result.get('response', '')
        
        clean_text = raw_text.replace('```json', '').replace('```', '').strip()
        
        try:
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
