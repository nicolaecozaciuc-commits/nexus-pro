import os
import re
import json
import requests
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURARE GLOBALA ---
PRODUCTS_DB = []
WORD_FREQ = Counter()

# SINONIME pentru domeniul instalatii sanitare/termice
SYNONYMS = {
    # Produse principale
    'CALORIFER': ['RADIATOR', 'ELEMENT', 'CORP'],
    'RADIATOR': ['CALORIFER', 'ELEMENT', 'CORP'],
    'SCARITA': ['RADIATOR BAIE', 'PORTPROSOP'],
    'SCARA': ['RADIATOR BAIE', 'PORTPROSOP'],
    'TEAVA': ['TIANA', 'TIGANA', 'CONDUCTA', 'TUB'],
    'TIANA': ['TEAVA', 'TIGANA', 'CONDUCTA'],
    'TIGANA': ['TEAVA', 'TIANA'],
    'ROBINET': ['VANA', 'VENTIL'],
    'ROSET': ['ROBINET'],
    'ROBICT': ['ROBINET'],
    'BOILER': ['BOLER', 'REZERVOR', 'ACM'],
    'BOLER': ['BOILER', 'REZERVOR'],
    
    # Materiale
    'OTEL': ['FE', 'FIER', 'METAL'],
    'ALAMA': ['BRONZ'],
    'CUPRU': ['CU'],
    'PPR': ['POLIPROPILENA'],
    'PEX': ['PEXA', 'PE'],
    'ZINC': ['ZINT', 'ZINCAT', 'ZN'],
    'ZINT': ['ZINC', 'ZINCAT', 'ZN'],
    'ZN': ['ZINC', 'ZINT', 'ZINCAT'],
    
    # Dimensiuni - TOL = 1 inch
    'TOL': ['INCH', 'TOLI'],
    
    # Tipuri conexiuni
    'MF': ['M/F'],
    'FF': ['F/F'],
    'MM': ['M/M'],
    'FE': ['FILET EXTERIOR', 'TATA'],
    'FI': ['FILET INTERIOR', 'MAMA'],
    
    # Accesorii
    'COT': ['COLT', 'CURBA'],
    'TEU': ['T', 'RAMIFICATIE'],
    'MUFA': ['CUPLAJ'],
    'NIPLU': ['NIPEL'],
    'REDUCTIE': ['REDUS', 'REDUCER', 'REDUCERE'],
    'REDUS': ['REDUCTIE', 'REDUCER'],
    'ADAPTOR': ['ADAPTO', 'RACORD', 'CONECTOR'],
    'RACORD': ['NACORD', 'RACOR'],
    'NACORD': ['RACORD'],
    'OLANDEZ': ['OLAND', 'HOLENDER'],
    'PRELUNGITOR': ['PRELUNGITO', 'EXTENSIE'],
    'BRATARA': ['BRATARI', 'COLIER', 'CLEMA'],
    'BRATARI': ['BRATARA', 'COLIER', 'CLEMA'],
    'CLEMA': ['BRATARA', 'BRATARI', 'COLIER'],
    'DOP': ['CAPAC', 'DOPI'],
    'SUPAPA': ['SUPAA', 'VALVA', 'CLAPETA'],
    'CLAPETA': ['SUPAPA', 'VALVA'],
    'AERISITOR': ['AERISTO', 'DEZAERISITOR'],
    'FILTRU': ['FILRU', 'FILTER'],
    
    # Altele
    'SERPENTINA': ['SERPENTIN', 'SCHIMBATOR'],
    'DUBLU': ['DUBLA', 'DOUBLE'],
    'TERMOSTATIC': ['TERMOSTAT', 'TERMO'],
    'TERMOSTAT': ['TERMOSTATIC', 'TERMO'],
    'GOLIRE': ['GOLIR', 'EVACUARE', 'VIDANJARE'],
    'SENS': ['DIRECTIE'],
    'VAS': ['RECIPIENT', 'EXPANSIUNE'],
    'POMPA': ['CIRCULATIE', 'RECIRCULARE'],
    'SCAUN': ['SCAUNEL', 'SUPORT'],
    'XILO': ['WILO', 'CIRCULATIE'],
    'WILO': ['XILO', 'CIRCULATIE'],
    
    # Erori OCR comune
    'DUSAR': ['DUSER', 'DUS'],
    'BLUZT': ['BULZI', 'BULZ'],
    'DFIER': ['FIER', 'OTEL'],
}

# Construim reverse lookup pentru sinonime
ALL_SYNONYMS = {}
for key, values in SYNONYMS.items():
    ALL_SYNONYMS[key] = set(values + [key])
    for v in values:
        if v not in ALL_SYNONYMS:
            ALL_SYNONYMS[v] = set()
        ALL_SYNONYMS[v].add(key)
        ALL_SYNONYMS[v].update(values)

def normalize_text(text):
    """Normalizeaza textul pentru cautare"""
    text = text.upper()
    # Inlocuieste caractere speciale cu spatiu
    text = re.sub(r'[X/\-\"\'\.\,\(\)\°]', ' ', text)
    # Normalizeaza dimensiuni comune
    text = re.sub(r'1\s*1\s*/\s*2', '1 1/2', text)
    text = re.sub(r'1\s*/\s*2', '1/2', text)
    text = re.sub(r'3\s*/\s*4', '3/4', text)
    text = re.sub(r'1\s*/\s*4', '1/4', text)
    # Elimina spatii multiple
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_search_words(text):
    """Obtine cuvintele din text inclusiv sinonimele"""
    norm = normalize_text(text)
    words = set()
    for w in norm.split():
        if len(w) >= 1:
            words.add(w)
            # Adauga si sinonime
            if w in ALL_SYNONYMS:
                words.update(ALL_SYNONYMS[w])
    return words

def load_database():
    """
    Incarca baza de date la pornirea serverului.
    Cauta automat fisiere .csv sau .xlsx in folderul curent.
    """
    global PRODUCTS_DB, WORD_FREQ
    print("🔄 Initializez incarcarea bazei de date...")
    
    # Cautam fisiere posibile
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx'))]
    file_path = files[0] if files else 'produse_nexus.csv'
    
    if not os.path.exists(file_path):
        print(f"⚠️ ATENTIE: Nu am gasit fisierul '{file_path}'. Urca-l pe server!")
        return

    try:
        # Citire inteligenta (Excel sau CSV)
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, dtype=str)
        else:
            try:
                df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip', engine='python')
            except:
                df = pd.read_csv(file_path, sep=';', dtype=str, on_bad_lines='skip', engine='python')

        df.columns = [c.strip().lower() for c in df.columns]
        
        col_den = next((c for c in df.columns if 'denumire' in c), None)
        col_cod = next((c for c in df.columns if c == 'cod'), None)
        col_sel = next((c for c in df.columns if 'selectie' in c), None)
        
        if not col_den:
            df = df.iloc[:, [0, 3, 12]]
            df.columns = ['cod_lung', 'denumire', 'cod_scurt']
        else:
            rename_map = {col_den: 'denumire'}
            if col_cod: rename_map[col_cod] = 'cod_lung'
            if col_sel: rename_map[col_sel] = 'cod_scurt'
            df = df.rename(columns=rename_map)

        df = df.fillna('')
        
        # Procesare cu indexare pentru cautare smart
        clean_data = []
        WORD_FREQ.clear()
        
        for _, row in df.iterrows():
            den = str(row.get('denumire', '')).strip()
            if len(den) < 2 or den.lower() == 'denumire': continue
            
            c_lung = str(row.get('cod_lung', '')).strip()
            c_scurt = str(row.get('cod_scurt', '')).strip()
            final_code = c_scurt if c_scurt else c_lung
            
            # Normalizare pentru cautare (cu sinonime)
            words = get_search_words(den)
            den_norm = normalize_text(den)
            
            # Actualizam frecventa cuvintelor
            for w in words:
                WORD_FREQ[w] += 1
            
            clean_data.append({
                'd': den,
                'c': final_code,
                'words': words,
                'norm': den_norm
            })
            
        PRODUCTS_DB = clean_data
        print(f"✅ SUCCES: {len(PRODUCTS_DB)} produse incarcate in memorie.")
        print(f"📊 Index: {len(WORD_FREQ)} cuvinte unice indexate.")
        
    except Exception as e:
        print(f"❌ EROARE CRITICA la citirea bazei de date: {e}")

# Încărcăm baza la start
load_database()

# --- RUTE WEB ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """
    API endpoint pentru cautarea SMART cu sinonime.
    Primeste JSON: { "query": "robinet 1/2" }
    Returneaza JSON: [ { "d": "Robinet...", "c": "123" }, ... ]
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query or len(query) < 2:
            return jsonify([])
        
        # Obtine cuvintele din query (cu sinonime)
        query_words = get_search_words(query)
        if not query_words:
            return jsonify([])
        
        results = []
        total = len(PRODUCTS_DB)
        
        for prod in PRODUCTS_DB:
            matched_words = query_words & prod['words']
            if not matched_words:
                continue
            
            # Calcul scor
            score = 0
            for w in matched_words:
                # IDF: cuvinte rare = scor mare
                freq = WORD_FREQ.get(w, 1)
                idf = total / freq
                score += idf
            
            # Bonus pentru mai multe potriviri
            match_ratio = len(matched_words) / len(query_words)
            score *= (1 + match_ratio)
            
            # Bonus daca potriveste numere exacte
            query_nums = set(re.findall(r'\d+', query.upper()))
            prod_nums = set(re.findall(r'\d+', prod.get('norm', '')))
            num_matches = len(query_nums & prod_nums)
            if num_matches > 0:
                score *= (1 + num_matches * 0.3)
            
            results.append({
                'd': prod['d'],
                'c': prod['c'],
                'score': score,
                'ratio': match_ratio
            })
        
        # Sorteaza dupa ratio apoi scor
        results.sort(key=lambda x: (-x['ratio'], -x['score']))
        
        # REGULA SPECIALA 1: Radiatoare OTEL - prioritate TERMO+ si 22K
        query_upper = query.upper()
        is_radiator_query = ('RADIATOR' in query_upper or 'CALORIFER' in query_upper) and 'OTEL' in query_upper
        is_scarita_query = 'SCARIT' in query_upper or 'SCARA' in query_upper or ('BAIE' in query_upper and ('RADIATOR' in query_upper or '600' in query_upper))
        
        if is_radiator_query or is_scarita_query:
            # Verifica daca NU e specificat 11 sau 33
            has_11 = '11K' in query_upper or ' 11 ' in query_upper or query_upper.endswith(' 11')
            has_33 = '33K' in query_upper or ' 33 ' in query_upper or query_upper.endswith(' 33')
            
            def is_termo_plus(denumire):
                return 'TERMO+' in denumire.upper() or 'TERMO +' in denumire.upper()
            
            def is_22k(denumire):
                d = denumire.upper()
                return '22K' in d or ' 22/' in d or '/22/' in d or 'TIP 22' in d or ' 22 ' in d
            
            # Prioritate: 1. TERMO+ cu 22K, 2. TERMO+, 3. 22K, 4. restul
            results_termo_22k = [r for r in results if is_termo_plus(r['d']) and (is_22k(r['d']) or not has_11 and not has_33)]
            results_termo = [r for r in results if is_termo_plus(r['d']) and r not in results_termo_22k]
            results_22k = [r for r in results if is_22k(r['d']) and not is_termo_plus(r['d']) and not has_11 and not has_33]
            results_other = [r for r in results if r not in results_termo_22k and r not in results_termo and r not in results_22k]
            
            results = results_termo_22k + results_termo + results_22k + results_other
        
        # Returneaza doar d si c (fara scor)
        return jsonify([{'d': r['d'], 'c': r['c']} for r in results[:30]])
        
    except Exception as e:
        print(f"Eroare search: {e}")
        return jsonify([])

@app.route('/api/ocr', methods=['POST'])
def ocr():
    """
    API endpoint pentru OCR cu Ollama LLaVA local.
    Primește JSON: { "image": "base64_string" }
    Returnează JSON: { "items": [{ "text": "...", "qty": 1 }] }
    """
    try:
        data = request.json
        image_base64 = data.get('image', '')
        
        if not image_base64:
            return jsonify({"error": "No image provided"})
        
        # Call Ollama API
        ollama_response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json={
                "model": "llava:7b",
                "prompt": "Extrage produsele din această imagine. Răspunde DOAR cu JSON valid, fără alte explicații. Format exact: { \"items\": [{ \"text\": \"nume produs\", \"qty\": 1 }] }. Dacă vezi cantități, include-le. Dacă nu vezi cantitate, pune qty: 1.",
                "images": [image_base64],
                "stream": False
            },
            timeout=120
        )
        
        result = ollama_response.json()
        response_text = result.get('response', '')
        
        # Clean and parse JSON
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return jsonify(parsed)
        else:
            return jsonify({"error": "Nu am putut extrage produse din imagine", "raw": response_text})
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Timeout - imaginea durează prea mult"})
    except Exception as e:
        print(f"Eroare OCR: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Configurare pentru rulare locală sau server
    app.run(host='0.0.0.0', port=8082, debug=True)
