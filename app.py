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
RULAJ_DB = {}  # Dictionar cu rulaj produse: {cod: cantitate_vanduta}

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
    'ALAMA': ['BRONZ', 'BRASS'],
    'BRONZ': ['ALAMA', 'BRASS'],
    'CANEA': ['MANETA', 'MANER'],
    'MANETA': ['CANEA', 'MANER'],
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
    'SCAUN': ['SCAUNEL', 'SUPORT', 'SCARITA', 'RADIATOR BAIE'],
    'XILO': ['WILO', 'HILO', 'CIRCULATIE'],
    'WILO': ['XILO', 'HILO', 'CIRCULATIE'],
    'HILO': ['WILO', 'XILO', 'CIRCULATIE'],
    
    # Erori OCR comune
    'DUSAR': ['DUSER', 'DUS'],
    'BLUZT': ['BULZI', 'BULZ'],
    'DFIER': ['FIER', 'OTEL'],
    
    # Abrevieri comune (adaugat din v8)
    'AUTOCUR': ['AUTOCURATIRE', 'AUTOCURATARE'],
    'AUTOCURATIRE': ['AUTOCUR'],
    'IGENIC': ['IGIENIC'],
    'IGIENIC': ['IGENIC'],
    'TERMOVENTIL': ['VTC', 'VENTIL TERMIC'],
    'VASE': ['VAS'],
    'EXP': ['EXPANSIUNE'],
    'EXPANSIUNE': ['EXP'],
    'CONECTOR': ['RACORD', 'ADAPTOR', 'CONECTORI'],
    'CONECTORI': ['RACORD', 'RACORDURI', 'CONECTOR'],
    'PARDOSEALA': ['PARDOSEA', 'INCALZIRE PODEA'],
    'PARDOSEA': ['PARDOSEALA'],
    
    # Vase expansiune
    'ALBASTRU': ['HIDROFOR', 'APA RECE', 'APA'],
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
            
            # Adaugam si codul in cuvintele de cautare
            if final_code:
                code_words = get_search_words(final_code)
                words.update(code_words)
            
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

# --- INCARCARE RULAJ PRODUSE ---
def load_rulaj():
    """Incarca rulajul produselor din fisierul Excel"""
    global RULAJ_DB
    import os
    
    # Cauta fisierul cu rulaj
    rulaj_paths = [
        'produse_vandute_2025.xls',
        'produse_vandute_2025_xlsx.xls',
        '/root/nexus-pro/produse_vandute_2025.xls',
        '/root/nexus-pro/produse_vandute_2025_xlsx.xls',
    ]
    
    rulaj_file = None
    for path in rulaj_paths:
        if os.path.exists(path):
            rulaj_file = path
            break
    
    if not rulaj_file:
        print("⚠️ Fisierul cu rulaj nu a fost gasit. Rulaj dezactivat.")
        return
    
    try:
        import pandas as pd
        df = pd.read_excel(rulaj_file)
        df['cantitate_abs'] = df['cantitate'].abs()
        
        # Grupam pe cod si sumam cantitatile
        df_grouped = df.groupby('cod_ext')['cantitate_abs'].sum().reset_index()
        
        for _, row in df_grouped.iterrows():
            cod = str(row['cod_ext']).strip()
            if cod:
                RULAJ_DB[cod] = row['cantitate_abs']
        
        print(f"📊 RULAJ: {len(RULAJ_DB)} produse cu rulaj incarcat.")
        
        # Top 5 pentru verificare
        top5 = sorted(RULAJ_DB.items(), key=lambda x: -x[1])[:5]
        for cod, cant in top5:
            print(f"   🔥 {cod}: {cant:.0f} buc")
            
    except Exception as e:
        print(f"⚠️ Eroare la incarcarea rulajului: {e}")

# Incarcam rulajul
load_rulaj()

# --- REGULA SPECIALA 18: ECHIPAMENTE PRINCIPALE - CANTITATE DEFAULT 1 ---
MAIN_EQUIPMENT_KEYWORDS = [
    'CENTRALA', 'CENTRALE', 'PUFFER', 'VAS EXPANSIUNE', 'VAS EXPAN',
    'BOILER', 'POMPA DE CALDURA', 'POMPA CALDURA', 'ACUMULATOR',
    'REZERVOR', 'SCHIMBATOR', 'BATERIE TERMICA'
]

def is_main_equipment(line_text):
    """
    Verifică dacă linia conține un echipament principal.
    Echipamentele principale sunt categorii mari (centrale, puffere, vase)
    nu accesorii (racorduri, robinete, etc).
    """
    line_upper = line_text.upper()
    return any(keyword in line_upper for keyword in MAIN_EQUIPMENT_KEYWORDS)

def extract_quantity_smart(line_text):
    """
    Extrage cantitate cu logică specială pentru echipamente principale.
    
    REGULA:
    - Pentru echipamente principale (CENTRALA, PUFFER, VAS EXPANSIUNE):
      Caută DOAR cantitate la final cu 'buc/bucati/bucăți'
      Dacă nu găsește → DEFAULT = 1
    
    - Pentru accesorii normale:
      Logica clasică - extrage ultimul număr sau primul număr rezonabil
    
    Exemple problematice FIXATE:
    - "Puffer 500l 2 serpentine" → 1 (nu 500!)
    - "Vas expansiune 50 albastru" → 1 (nu 50!)
    - "Centrala 28-30kw" → 1 (nu 28!)
    - "Puffer 500l 2 serpentine 3 buc" → 3 (corect!)
    """
    line_text = line_text.strip()
    
    # === FIX v20: Elimină dimensiuni cu litri (50L, 100L) înainte de procesare ===
    # Pentru a evita confuzia între "Vas 50L" (dimensiune) și "50 buc" (cantitate)
    line_text_clean = re.sub(r'\d+\s*[Ll]\b', '', line_text)
    
    # ECHIPAMENTE PRINCIPALE - cantitate doar cu "buc" explicit
    if is_main_equipment(line_text_clean):
        # Caută pattern "X buc" la final
        match_buc = re.search(r'(\d+)\s*(buc|bucati|bucăți|bucată)\s*$', 
                             line_text_clean, re.IGNORECASE)
        if match_buc:
            return int(match_buc.group(1))
        else:
            # DEFAULT pentru echipamente principale
            return 1
    
    # ACCESORII - logica normală
    else:
        # Caută pattern "X buc" oriunde
        match_buc = re.search(r'(\d+)\s*(buc|bucati|bucăți|bucată)', 
                             line_text_clean, re.IGNORECASE)
        if match_buc:
            return int(match_buc.group(1))
        
        # Caută ultimul număr din linie (probabil cantitatea)
        all_numbers = re.findall(r'\d+', line_text_clean)
        if all_numbers:
            # Returnează ultimul număr găsit
            return int(all_numbers[-1])
        
        # Default general
        return 1

# --- RUTE WEB ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/parse-text', methods=['POST'])
def parse_text():
    """
    API endpoint nou pentru parsarea SAFE a textului paste-uit.
    Folosește REGULA SPECIALA 18 pentru cantități.
    
    Primește JSON: { "text": "1) Centrala...\n2) Puffer..." }
    Returnează JSON: { "items": [{ "text": "Centrala...", "qty": 1 }, ...] }
    """
    try:
        data = request.json
        raw_text = data.get('text', '').strip()
        
        if not raw_text:
            return jsonify({"items": []})
        
        # Parsare linii
        lines = raw_text.split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Elimină numerotarea de la început (1), 2., etc.)
            line_clean = re.sub(r'^\d+[\)\.]\s*', '', line)
            
            if not line_clean:
                continue
            
            # Extrage cantitate cu logica SMART (REGULA 18)
            qty = extract_quantity_smart(line_clean)
            
            # Elimină cantitatea din text pentru afișare
            text_without_qty = re.sub(r'\s*\d+\s*(buc|bucati|bucăți|bucată)\s*$', '', 
                                     line_clean, flags=re.IGNORECASE)
            
            items.append({
                "text": text_without_qty.strip(),
                "qty": qty
            })
        
        return jsonify({"items": items})
        
    except Exception as e:
        print(f"Eroare parse-text: {e}")
        return jsonify({"error": str(e)})

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
        
        # === FIX v25: NORMALIZARE DIACRITICE ROMÂNEȘTI ===
        # Problema: query-urile vin cu "peleți", "expansiunea", "espansiune"
        # Soluție: Normalizăm toate diacriticele la caractere simple
        query_normalized = query.upper()
        query_normalized = query_normalized.replace('Ț', 'T').replace('Ţ', 'T')
        query_normalized = query_normalized.replace('Ș', 'S').replace('Ş', 'S')
        query_normalized = query_normalized.replace('Ă', 'A').replace('Â', 'A').replace('Î', 'I')
        # Fix greșeli de scriere comune: "espansiune" → "expansiune"
        query_normalized = query_normalized.replace('ESPANSIUNE', 'EXPANSIUNE')
        query_normalized = query_normalized.replace('ESPANS', 'EXPANS')
        
        # === FIX v26: NORMALIZARE DIMENSIUNI PPR ===
        # PPR folosește dimensiuni standard: 20, 25, 32, 40, 50
        # "3ex1" trebuie interpretat ca "32x1", "2ex" ca "25", etc
        query_normalized = query_normalized.replace('5EX', '50X')
        query_normalized = query_normalized.replace('4EX', '40X')
        query_normalized = query_normalized.replace('3EX', '32X')
        query_normalized = query_normalized.replace('2EX', '25X')
        # Fix și pentru 1½ (1.5 inch)
        query_normalized = query_normalized.replace('1½', '1.5')
        query_normalized = query_normalized.replace('1 1/2', '1.5')
        
        # === LOG v24: Afișează TOATE query-urile relevante pentru debugging ===
        # Afișăm doar query-uri lungi (>15 caractere) care conțin cuvinte cheie
        if len(query) > 15:
            query_check = query_normalized  # Folosim query normalizat
            if 'CENTRAL' in query_check or 'PELETI' in query_check or 'VAS' in query_check or 'EXPAN' in query_check:
                print(f"📥 QUERY LUNG PRIMIT: '{query}' → NORMALIZAT: '{query_normalized}'")
                print(f"   Lungime: {len(query)} caractere")
        
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
            
            # Cautare substring in cod si denumire (pentru coduri ca R470AX003)
            # FIX v25: Folosim query_normalized în loc de query_normalized
            substring_match = False
            if len(query_normalized) >= 3:
                if query_normalized in prod['c'].upper() or query_normalized in prod['d'].upper():
                    substring_match = True
            
            if not matched_words and not substring_match:
                continue
            
            # Calcul scor
            score = 0
            for w in matched_words:
                # IDF: cuvinte rare = scor mare
                freq = WORD_FREQ.get(w, 1)
                idf = total / freq
                score += idf
            
            # Bonus pentru substring match in cod
            if substring_match:
                score += 100  # Prioritate mare pentru match exact in cod
            
            # Bonus pentru mai multe potriviri
            match_ratio = len(matched_words) / len(query_words) if matched_words else (0.5 if substring_match else 0)
            score *= (1 + match_ratio)
            
            # Bonus daca potriveste numere exacte
            query_nums = set(re.findall(r'\d+', query.upper()))
            prod_nums = set(re.findall(r'\d+', prod.get('norm', '')))
            num_matches = len(query_nums & prod_nums)
            if num_matches > 0:
                score *= (1 + num_matches * 0.3)
            
            # Bonus pentru produse cu rulaj mare (cele mai vandute)
            cod_produs = prod['c']
            rulaj = RULAJ_DB.get(cod_produs, 0)
            if rulaj > 10000:
                score *= 1.5  # Bonus 50% pentru produse foarte vandute
            elif rulaj > 1000:
                score *= 1.3  # Bonus 30% pentru produse vandute
            elif rulaj > 100:
                score *= 1.1  # Bonus 10% pentru produse cu rulaj
            
            # Bonus pentru dimensiuni EXACTE (ex: 600x600 gaseste 600X600 primul)
            # Fix: elimina punctele din numere (1.000 -> 1000)
            query_for_dims = re.sub(r'(\d)\.(\d)', r'\1\2', query.upper())
            query_nums_list = re.findall(r'\d+', query_for_dims)
            if len(query_nums_list) >= 2:
                query_dims = ''.join(query_nums_list)  # "600600"
                prod_dims = re.sub(r'[^0-9]', '', prod['d'].upper())  # extrage toate numerele
                if query_dims in prod_dims:
                    score *= 3  # Bonus mare pentru match exact dimensiuni
            
            # === FIX v20: BOOST MASIV pentru centrale peleți ===
            # === FIX v21: Cresc BOOST de la x20 la x100 ===
            # Când utilizatorul caută "centrala peleti", sistemul trebuie să prioritizeze
            # centrale pe peleți față de alte produse (ex: kit-uri GPL) chiar dacă acestea
            # au match mai bun pe dimensiuni
            if 'CENTRALA' in query_normalized and 'PELETI' in query_normalized:
                prod_upper = prod['d'].upper()
                if 'CENTRALA' in prod_upper and 'PELETI' in prod_upper:
                    score *= 100  # x100 pentru a GARANTA că depășește orice alt match
            
            results.append({
                'd': prod['d'],
                'c': prod['c'],
                'score': score,
                'ratio': match_ratio
            })
        
        # Sorteaza dupa ratio apoi scor
        results.sort(key=lambda x: (-x['ratio'], -x['score']))
        
        # REGULA SPECIALA 1: Radiatoare OTEL - prioritate TERMO+ si 22K
        # FIX v25: query_normalized e deja definit la început cu normalizare diacritice
        is_radiator_query = ('RADIATOR' in query_normalized or 'CALORIFER' in query_normalized) and 'OTEL' in query_normalized
        is_scarita_query = 'SCARIT' in query_normalized or 'SCARA' in query_normalized or ('BAIE' in query_normalized and ('RADIATOR' in query_normalized or '600' in query_normalized))
        
        if is_radiator_query or is_scarita_query:
            # Verifica daca NU e specificat 11 sau 33
            has_11 = '11K' in query_normalized or ' 11 ' in query_normalized or query_normalized.endswith(' 11')
            has_33 = '33K' in query_normalized or ' 33 ' in query_normalized or query_normalized.endswith(' 33')
            
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
        
        # REGULA SPECIALA 2: Vase expansiune - prioritate VR/VRV pentru boiler/centrala, VAV/VAO pentru hidrofor
        if 'VAS' in query_normalized and ('EXPANSIUNE' in query_normalized or 'EXPAN' in query_normalized):
            has_boiler = 'BOILER' in query_normalized or 'CENTRALA' in query_normalized or 'TERMIC' in query_normalized
            has_hidrofor = 'HIDROFOR' in query_normalized or 'POMPARE' in query_normalized or 'APA' in query_normalized
            
            def get_vas_priority(cod, denumire):
                c = cod.upper()
                d = denumire.upper()
                
                if has_boiler:
                    # Pentru boiler/centrala: VR prioritar
                    if c.startswith('VR') and not c.startswith('VRV'):
                        return 1
                    if c.startswith('VRV'):
                        return 2
                    return 10
                elif has_hidrofor:
                    # Pentru hidrofor: VAV prioritar
                    if c.startswith('VAV'):
                        return 1
                    if c.startswith('VAO'):
                        return 2
                    return 10
                else:
                    # Fara specificatie: general VR > VRV > VAV > VAO
                    if c.startswith('VR') and not c.startswith('VRV'):
                        return 1
                    if c.startswith('VRV'):
                        return 2
                    if c.startswith('VAV'):
                        return 3
                    if c.startswith('VAO'):
                        return 4
                    return 10
            
            results.sort(key=lambda r: (get_vas_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 3: Pompe WILO/HILO/XILO - YONOS PICO 1.0 prioritar
        if any(x in query_normalized for x in ['WILO', 'HILO', 'XILO', 'POMPA', 'CIRCULATIE']):
            def get_pompa_priority(cod, denumire):
                d = denumire.upper()
                # YONOS PICO 1.0 prioritar
                if 'YONOS PICO 1.0' in d or 'YONOS PICO 10' in d:
                    return 1
                if 'YONOS PICO' in d:
                    return 2
                if 'YONOS' in d:
                    return 3
                if any(x in d for x in ['WILO', 'HILO', 'XILO']):
                    return 4
                return 10
            results.sort(key=lambda r: (get_pompa_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 4: Cand cauta explicit "TERMO+" -> prioritar TERMO+
        if 'TERMO+' in query_normalized or 'TERMO +' in query_normalized:
            def has_termo_plus(denumire):
                d = denumire.upper()
                return 'TERMO+' in d or 'TERMO +' in d
            results.sort(key=lambda r: (0 if has_termo_plus(r['d']) else 1, -r.get('score', 0)))
        
        # REGULA SPECIALA 5: PUFFER - TERMO+ cu METAL+INOX prioritar
        if 'PUFFER' in query_normalized:
            def get_puffer_priority(cod, denumire):
                d = denumire.upper()
                c = cod.upper()
                # HPh* = PUFFER TERMO+ cu METAL+INOX
                if 'TERMO+' in d and 'METAL' in d and 'INOX' in d:
                    return 1
                if c.startswith('HPh'):
                    return 2
                if 'TERMO+' in d:
                    return 3
                if 'METAL' in d and 'INOX' in d:
                    return 4
                return 10
            results.sort(key=lambda r: (get_puffer_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 6: KIT TUR-RETUR - R470X001 prioritar
        if ('KIT' in query_normalized or 'SET' in query_normalized) and 'TUR' in query_normalized and 'RETUR' in query_normalized:
            def get_kit_tur_retur_priority(cod, denumire):
                c = cod.upper()
                if 'R470X001' in c or 'R470AX001' in c:
                    return 1
                if c.startswith('R470'):
                    return 2
                return 10
            results.sort(key=lambda r: (get_kit_tur_retur_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 7: GRUP SOLAR/POMPARE SOLAR - GPD212 GRUNDFOS prioritar
        if 'GRUP' in query_normalized and ('SOLAR' in query_normalized or 'POMPARE' in query_normalized):
            def get_grup_solar_priority(cod, denumire):
                c = cod.upper()
                d = denumire.upper()
                # GPD212 = Grup pompare solar Grundfos prioritar
                if 'GPD212' in c:
                    return 1
                if 'GRUNDFOS' in d and 'SOLAR' in d:
                    return 2
                return 10
            results.sort(key=lambda r: (get_grup_solar_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 8: KIT AERISITOR SOLAR - 63280648 cu pipa prioritar
        if ('KIT' in query_normalized or 'SET' in query_normalized) and 'AERISITOR' in query_normalized and 'SOLAR' in query_normalized:
            def get_kit_aerisitor_priority(cod, denumire):
                c = cod.upper()
                d = denumire.upper()
                # 63280648 = Kit aerisitor solar cu pipa prioritar
                if '63280648' in c:
                    return 1
                if 'PIPA' in d or 'TIJA' in d:
                    return 2
                return 10
            results.sort(key=lambda r: (get_kit_aerisitor_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 9: VAS SOLAR - VS* prioritar
        if 'VAS' in query_normalized and 'SOLAR' in query_normalized:
            def get_vas_solar_priority(cod, denumire):
                c = cod.upper()
                # VS12, VS18, VS24 = Vase solare prioritare
                if c.startswith('VS') and any(x in c for x in ['12', '18', '24', '35', '50']):
                    return 1
                if c.startswith('VS'):
                    return 2
                return 10
            results.sort(key=lambda r: (get_vas_solar_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 10: SET CONECTORI INOX 16 - 63281189 prioritar
        if ('SET' in query_normalized or 'KIT' in query_normalized) and 'CONECTOR' in query_normalized and ('16' in query or 'INOX' in query_normalized):
            def get_set_conectori_priority(cod, denumire):
                c = cod.upper()
                # 63281189 = Set conectori inox 16 prioritar
                if '63281189' in c:
                    return 1
                if 'INOX' in denumire.upper() and '16' in cod:
                    return 2
                return 10
            results.sort(key=lambda r: (get_set_conectori_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 11: GRUP PARDOSEALA - OTR-WP/OTF-WP/OZR-WP TERMO+ prioritar
        if 'GRUP' in query_normalized and ('PARDOSEALA' in query_normalized or 'PARDOSEA' in query_normalized):
            def get_grup_pardoseala_priority(cod, denumire):
                c = cod.upper()
                d = denumire.upper()
                # OTR-WP, OTF-WP, OZR-WP = Grupuri pardoseala Termo+ prioritare
                if any(x in c for x in ['OTR-WP', 'OTF-WP', 'OZR-WP']):
                    return 1
                if c.startswith('OT') or c.startswith('OZ'):
                    return 2
                if 'TERMO+' in d:
                    return 4
                return 10
            results.sort(key=lambda r: (get_grup_pardoseala_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 12: TERMOMANOMETRU - THMTAA prioritar
        if 'TERMOMANOMETRU' in query_normalized or 'TERMO MANOMETRU' in query_normalized:
            def get_termomanometru_priority(cod, denumire):
                c = cod.upper()
                # THMTAA = Termomanometru axial prioritar
                if c.startswith('THMTAA'):
                    return 1
                if c.startswith('THMTA'):
                    return 2
                return 10
            results.sort(key=lambda r: (get_termomanometru_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 13: ROBINET CU CANEA/MANETA - ASTER prioritar
        if 'ROBINET' in query_normalized and ('CANEA' in query_normalized or 'MANETA' in query_normalized):
            def get_robinet_canea_priority(cod, denumire):
                d = denumire.upper()
                # ASTER prioritar daca nu specifica FF sau MF
                if 'ASTER' in d:
                    return 1
                if 'FF' in d or 'MF' in d:
                    return 2
                return 10
            results.sort(key=lambda r: (get_robinet_canea_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 14: OLANDEZ BRONZ/ALAMA - OLDR* prioritar
        if 'OLANDEZ' in query_normalized and ('BRONZ' in query_normalized or 'ALAMA' in query_normalized):
            def get_olandez_priority(cod, denumire):
                c = cod.upper()
                # OLDR = Olandez drept prioritar
                if c.startswith('OLDR'):
                    return 1
                return 10
            results.sort(key=lambda r: (get_olandez_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 15: FILTRU Y - Y* ECO prioritar
        if 'FILTRU' in query_normalized and 'Y' in query_normalized:
            def get_filtru_y_priority(cod, denumire):
                c = cod.upper()
                d = denumire.upper()
                # Y114 etc = Filtru Y ECO prioritar
                if c.startswith('Y') and 'FILTRU' in d and 'Y' in d:
                    return 1
                if 'ECO' in d:
                    return 2
                return 10
            results.sort(key=lambda r: (get_filtru_y_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 16: AERISITOR AUTOMAT GIACOMINI - R88IY003 prioritar
        if 'AERISITOR' in query_normalized and 'GIACOMINI' in query_normalized:
            def get_aerisitor_giacomini_priority(cod, denumire):
                c = cod.upper()
                # R88IY003 = Aerisitor automat de coloana Giacomini prioritar
                if 'R88IY' in c:
                    return 1
                if 'GIACOMINI' in denumire.upper():
                    return 2
                return 10
            results.sort(key=lambda r: (get_aerisitor_giacomini_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 17: SUPORT VAS - SUPORT PERETE VASE EXP prioritar
        if 'SUPORT' in query_normalized and 'VAS' in query_normalized:
            def get_suport_vas_priority(cod, denumire):
                d = denumire.upper()
                c = cod.upper()
                # SUPORT PERETE VASE EXP prioritar
                if 'PERETE' in d and 'VAS' in d:
                    return 1
                if c.startswith('2068') or c.startswith('2069'):
                    return 2
                return 10
            results.sort(key=lambda r: (get_suport_vas_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 18: CENTRALA PELETI - autocuratire prioritar
        if 'CENTRALA' in query_normalized and 'PELETI' in query_normalized:
            def get_centrala_peleti_priority(cod, denumire):
                d = denumire.upper()
                # AUTOCURATIRE prioritar
                if 'AUTOCURATIRE' in d or 'AUTOCUR' in d:
                    return 1
                if 'PELETI' in d and 'CENTRALA' in d:
                    return 2
                return 10
            results.sort(key=lambda r: (get_centrala_peleti_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 19: NIPLU/REDUCTIE mari (>=1½") - ZN (zincate) prioritar
        if ('NIPLU' in query_normalized or 'REDUCTIE' in query_normalized or 'REDUS' in query_normalized):
            # Verifică dacă sunt dimensiuni mari (1½" sau 2")
            has_large_dim = any(x in query_normalized for x in ['1 1/2', '1½', '1 1 2', '2"', '2 ', ' 2'])
            # Verifică dacă NU specifică alama/eco
            has_alama = 'ALAMA' in query_normalized or 'BRONZ' in query_normalized
            has_eco = 'ECO' in query_normalized
            
            if has_large_dim and not has_alama and not has_eco:
                def get_niplu_zn_priority(cod, denumire):
                    d = denumire.upper()
                    c = cod.upper()
                    # ZN = zinc/zincate prioritar pentru dimensiuni mari
                    if 'ZN' in c or 'ZINC' in d or 'ZINCAT' in d:
                        return 1
                    if 'ALAMA' in d or 'BRONZ' in d:
                        return 3
                    return 2
                results.sort(key=lambda r: (get_niplu_zn_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 20: VAS EXPANSIUNE - VRV > VR > VRW (rafinare pentru vase roșii)
        if 'VAS' in query_normalized and ('EXPANSIUNE' in query_normalized or 'EXPAN' in query_normalized):
            has_hidrofor = 'HIDROFOR' in query_normalized or 'APA' in query_normalized
            
            if not has_hidrofor:
                # Pentru încălzire/generale: VRV > VR > VRW
                def get_vas_rosu_priority(cod):
                    c = cod.upper()
                    if c.startswith('VRV'):
                        return 1  # Vertical cu suport - CEL MAI FOLOSIT
                    if c.startswith('VR') and not c.startswith('VRV') and not c.startswith('VRW'):
                        return 2  # Normal
                    if c.startswith('VRW'):
                        return 3  # Orizontal
                    return 10
                
                results.sort(key=lambda r: (get_vas_rosu_priority(r['c']), -r.get('score', 0)))
        
        # REGULA SPECIALA 21: GRUP SIGURANTA - HERZ prioritar
        if 'GRUP' in query_normalized and ('SIGURANTA' in query_normalized or 'SIG' in query_normalized):
            def get_grup_siguranta_priority(cod, denumire):
                d = denumire.upper()
                c = cod.upper()
                # HERZ prioritar
                if 'HERZ' in d or 'HERZ' in c:
                    return 1
                if 'GRSIGHERZ' in c:
                    return 2
                return 10
            results.sort(key=lambda r: (get_grup_siguranta_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 22: VAS ALBASTRU/HIDROFOR - VAO/VAV prioritar peste INOX
        # FIX v21: Când caută "vas albastru" sau "vas hidrofor" → prioritate vase albastre (VAO/VAV)
        if 'VAS' in query_normalized and ('EXPANSIUNE' in query_normalized or 'EXPAN' in query_normalized):
            has_albastru = 'ALBASTRU' in query_normalized or 'HIDROFOR' in query_normalized or 'APA' in query_normalized
            
            if has_albastru:
                def get_vas_albastru_priority(cod, denumire):
                    c = cod.upper()
                    d = denumire.upper()
                    # VAO/VAV = vase albastre (hidrofor) prioritare
                    if c.startswith('VAO') or c.startswith('VAV'):
                        return 1
                    # Excludem INOX (care nu sunt vase expansiune)
                    if c.startswith('INOX') or 'INOX' in c[:10]:
                        return 10
                    return 5
                
                results.sort(key=lambda r: (get_vas_albastru_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 23: ROBINET CU OLANDEZ - EFCO/COLLETTORE prioritar
        # FIX v26: Pentru robinete cu olandez → prioritate EFCO și COLLETTORE
        if 'ROBINET' in query_normalized and 'OLANDEZ' in query_normalized:
            def get_robinet_olandez_priority(cod, denumire):
                c = cod.upper()
                d = denumire.upper()
                # EFCO prioritar
                if 'EFCO' in c or 'EFCO' in d:
                    return 1
                # COLLETTORE prioritar
                if 'COLLETTORE' in d:
                    return 2
                return 10
            results.sort(key=lambda r: (get_robinet_olandez_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 24: PPR FARA CULOARE - ALB prioritar
        # FIX v26: Când caută PPR fără să specifice culoare → prioritate ALB (exclude gri/verde)
        if 'PPR' in query_normalized:
            has_verde = 'VERDE' in query_normalized
            has_gri = 'GRI' in query_normalized
            has_alb = 'ALB' in query_normalized
            
            # Dacă NU specifică culoare → prioritate ALB
            if not has_verde and not has_gri and not has_alb:
                def get_ppr_alb_priority(cod, denumire):
                    d = denumire.upper()
                    # ALB prioritar
                    if 'ALB' in d:
                        return 1
                    # Exclude VERDE și GRI
                    if 'VERDE' in d or 'GRI' in d:
                        return 10
                    # Neutru (fără culoare specificată în denumire)
                    return 5
                results.sort(key=lambda r: (get_ppr_alb_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 25: OLANDEZ PPR - când e singură dimensiune → caută XXxXX
        # FIX v26: "Olandez PPR 32" → caută 32x32, nu 32x1
        if 'OLANDEZ' in query_normalized and 'PPR' in query_normalized:
            # Extrage dimensiunea (20, 25, 32, 40, 50)
            import re
            ppr_dims = re.findall(r'\b(20|25|32|40|50)\b', query_normalized)
            if len(ppr_dims) == 1:
                dim = ppr_dims[0]
                # Caută produse cu XXxXX (ex: 32x32)
                def get_olandez_ppr_priority(cod, denumire):
                    c = cod.upper()
                    d = denumire.upper()
                    pattern = f'{dim}X{dim}'
                    if pattern in c or pattern in d or f'{dim}X{dim}' in d:
                        return 1
                    # Exclude dimensiuni diferite (ex: 32x1)
                    if f'{dim}X' in c or f'{dim}X' in d:
                        other_dim = re.search(f'{dim}X(\\d+)', c + d)
                        if other_dim and other_dim.group(1) != dim:
                            return 10
                    return 5
                results.sort(key=lambda r: (get_olandez_ppr_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 26: REDUCTIE/NIPLU REDUS - match precis dimensiuni
        # FIX v26: "Reducție 1½la 1" → 1.5"x1", nu 1.25"x1"
        if ('REDUCTIE' in query_normalized or 'REDUS' in query_normalized or 'NIPLU' in query_normalized):
            # Extrage toate dimensiunile din query
            import re
            # Caută pattern-uri precum 1.5, 1/2, 3/4, etc
            dims_in_query = set()
            # Dimensiuni cu punct (1.5)
            dims_in_query.update(re.findall(r'\d+\.\d+', query_normalized))
            # Dimensiuni întregi
            dims_in_query.update(re.findall(r'\b[1-4]\b', query_normalized))
            
            if len(dims_in_query) >= 2:
                # Are cel puțin 2 dimensiuni → match exact
                def get_reductie_priority(cod, denumire):
                    text = (cod + ' ' + denumire).upper()
                    # Verifică dacă TOATE dimensiunile din query apar în produs
                    matches = sum(1 for dim in dims_in_query if dim in text)
                    if matches == len(dims_in_query):
                        return 1
                    elif matches > 0:
                        return 5
                    return 10
                results.sort(key=lambda r: (get_reductie_priority(r['c'], r['d']), -r.get('score', 0)))
        
        # === FIX v22: FILTRĂRI FINALE pentru cazuri problemă ===
        
        # FILTRARE 1: CENTRALA PELETI - Exclude tot ce NU conține "peleti"
        if 'CENTRALA' in query_normalized and 'PELETI' in query_normalized:
            # === DEBUG v23: Afișează ce se întâmplă la filtrare ===
            print(f"🔍 DEBUG FILTRARE CENTRALA: Query='{query}'")
            print(f"   Înainte filtrare: {len(results)} produse")
            print(f"   Top 3 înainte:")
            for i, r in enumerate(results[:3], 1):
                print(f"      {i}. {r['c']}: {r['d'][:60]}")
            
            # Păstrează DOAR produse care au "PELETI" sau "PELET" în denumire
            results = [r for r in results if 'PELETI' in r['d'].upper() or 'PELET' in r['d'].upper()]
            
            # === DEBUG v23: Afișează rezultatul filtrării ===
            print(f"   După filtrare: {len(results)} produse")
            print(f"   Top 3 după:")
            for i, r in enumerate(results[:3], 1):
                print(f"      {i}. {r['c']}: {r['d'][:60]}")
        
        # FILTRARE 2: VAS ALBASTRU/HIDROFOR - Prioritizează VAO/VAV, exclude INOX
        if 'VAS' in query_normalized and ('EXPANSIUNE' in query_normalized or 'EXPAN' in query_normalized):
            has_albastru = 'ALBASTRU' in query_normalized or 'HIDROFOR' in query_normalized or 'APA' in query_normalized
            
            if has_albastru:
                # === DEBUG v23: Afișează ce se întâmplă la filtrare vase ===
                print(f"🔍 DEBUG FILTRARE VAS ALBASTRU: Query='{query}'")
                print(f"   Înainte filtrare: {len(results)} produse")
                print(f"   Top 3 înainte:")
                for i, r in enumerate(results[:3], 1):
                    print(f"      {i}. {r['c']}: {r['d'][:60]}")
                
                # Caută explicit vase albastre (VAO/VAV)
                vas_albastru = [r for r in results if r['c'].upper().startswith('VAO') or r['c'].upper().startswith('VAV')]
                
                if vas_albastru:
                    # Găsit vase albastre → folosește doar pe acestea
                    results = vas_albastru
                    print(f"   ✓ Găsit {len(vas_albastru)} vase albastre (VAO/VAV)")
                else:
                    # Fallback: exclude INOX (nu sunt vase expansiune)
                    results = [r for r in results if not r['c'].upper().startswith('INOX')]
                    print(f"   ✓ Nu găsit VAO/VAV, exclus INOX, rămas {len(results)} produse")
                
                # === DEBUG v23: Afișează rezultatul ===
                print(f"   După filtrare: {len(results)} produse")
                print(f"   Top 3 după:")
                for i, r in enumerate(results[:3], 1):
                    print(f"      {i}. {r['c']}: {r['d'][:60]}")
        
        # Returneaza doar d si c (fara scor)
        return jsonify([{'d': r['d'], 'c': r['c']} for r in results[:30]])
        
    except Exception as e:
        print(f"Eroare search: {e}")
        return jsonify([])

# --- CHEIA OPENAI (stocata pe server) ---
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Daca nu e in environment, citeste din fisier
if not OPENAI_API_KEY:
    try:
        with open('/root/nexus-pro/.openai_key', 'r') as f:
            OPENAI_API_KEY = f.read().strip()
    except:
        pass

@app.route('/api/ocr-openai', methods=['POST'])
def ocr_openai():
    """
    API endpoint pentru OCR cu OpenAI GPT-4o.
    Cheia API e stocata pe server - nu mai trebuie introdusa in browser.
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "OpenAI API key not configured on server"})
        
        data = request.json
        image_base64 = data.get('image', '')
        
        if not image_base64:
            return jsonify({"error": "No image provided"})
        
        # Call OpenAI API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o",
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extrage TOATE produsele din aceasta imagine. REGULI: 1) Extrage FIECARE linie care contine un produs, NU sari peste nicio linie. 2) Daca e o lista scrisa de mana sau tiparita, extrage TOATE elementele de la PRIMA pana la ULTIMA linie. 3) Daca vezi un titlu/categorie urmat de dimensiuni, combina-le. 4) Cantitatea poate fi 'buc', 'bucati', un numar, sau la sfarsitul liniei. 5) Daca nu vezi cantitate explicita, pune qty: 1. 6) Include CENTRALE, PUFFER, BOILER, POMPE - sunt produse importante! 7) ATENTIE LA CIFRE SCRISE DE MANA: 0 nu e 6, 8 nu e 9, 1 nu e 7, 00 nu e 06. Verifica de doua ori numerele! Raspunde DOAR cu JSON valid. Format: { \"items\": [{ \"text\": \"nume produs complet\", \"qty\": cantitate_numar }] }"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }],
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        result = response.json()
        
        if 'error' in result:
            return jsonify({"error": result['error'].get('message', 'OpenAI error')})
        
        content = result['choices'][0]['message']['content']
        content = content.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(content)
        
        return jsonify(parsed)
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "Timeout - procesarea durează prea mult"})
    except Exception as e:
        print(f"Eroare OCR OpenAI: {e}")
        return jsonify({"error": str(e)})

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
