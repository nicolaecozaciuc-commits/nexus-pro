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
            
            # Cautare substring in cod si denumire (pentru coduri ca R470AX003)
            query_upper = query.upper()
            substring_match = False
            if len(query_upper) >= 3:
                if query_upper in prod['c'].upper() or query_upper in prod['d'].upper():
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
            query_nums_list = re.findall(r'\d+', query.upper())
            if len(query_nums_list) >= 2:
                query_dims = ''.join(query_nums_list)  # "600600"
                prod_dims = re.sub(r'[^0-9]', '', prod['d'].upper())  # extrage toate numerele
                if query_dims in prod_dims:
                    score *= 3  # Bonus mare pentru match exact dimensiuni
            
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
        
        # REGULA SPECIALA 2: Vase expansiune - prioritate VR/VRV pentru boiler/centrala, VAV/VAO pentru hidrofor
        is_vas_query = 'VAS' in query_upper and ('LITRI' in query_upper or 'EXPAN' in query_upper or any(c.isdigit() for c in query_upper))
        
        if is_vas_query:
            is_hidrofor = 'HIDROFOR' in query_upper or 'APA RECE' in query_upper or 'APA' in query_upper
            is_incalzire = 'BOILER' in query_upper or 'CENTRALA' in query_upper or 'INCALZIRE' in query_upper or 'BOLER' in query_upper
            
            def get_vas_priority(denumire, cod):
                d = denumire.upper()
                c = cod.upper()
                # VR = vas rosu fara suport (prioritate maxima pentru incalzire)
                if c.startswith('VR') and not c.startswith('VRV'):
                    return 1
                # VRV = vas rosu vertical cu suport
                if c.startswith('VRV'):
                    return 2
                # VAV/VAO = vas albastru (pentru hidrofor)
                if c.startswith('VAV') or c.startswith('VAO'):
                    return 3 if is_hidrofor else 5
                return 10
            
            if is_incalzire or (not is_hidrofor):
                # Pentru boiler/centrala: VR > VRV > restul
                results.sort(key=lambda r: (get_vas_priority(r['d'], r['c']), -r.get('score', 0)))
            elif is_hidrofor:
                # Pentru hidrofor: VAV > VAO > restul
                def get_hidrofor_priority(denumire, cod):
                    c = cod.upper()
                    if c.startswith('VAV'):
                        return 1
                    if c.startswith('VAO'):
                        return 2
                    return 10
                results.sort(key=lambda r: (get_hidrofor_priority(r['d'], r['c']), -r.get('score', 0)))
        
        # REGULA SPECIALA 3: Pompe WILO/HILO - prioritate YONOS PICO 1.0
        is_pompa_query = ('POMPA' in query_upper or 'WILO' in query_upper or 'HILO' in query_upper or 'XILO' in query_upper)
        
        if is_pompa_query:
            def get_pompa_priority(denumire):
                d = denumire.upper()
                # YONOS PICO 1.0 = prioritate maxima
                if 'YONOS PICO 1.0' in d or 'YONOS PICO1.0' in d:
                    return 1
                # YONOS PICO = prioritate 2
                if 'YONOS PICO' in d:
                    return 2
                # YONOS = prioritate 3
                if 'YONOS' in d:
                    return 3
                # Alte WILO
                if 'WILO' in d:
                    return 5
                return 10
            
            results.sort(key=lambda r: (get_pompa_priority(r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 4: TERMO+ prioritar cand se cauta explicit termo+ (adaugat din v8)
        if 'TERMO+' in query_upper or 'TERMO +' in query_upper:
            def get_termo_priority(denumire):
                d = denumire.upper()
                if 'TERMO+' in d or 'TERMO +' in d:
                    return 1
                return 10
            results.sort(key=lambda r: (get_termo_priority(r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 5: PUFFER - prioritate TERMO+ cu METAL + INOX (adaugat din v8)
        is_puffer_query = 'PUFFER' in query_upper
        
        if is_puffer_query:
            def get_puffer_priority(denumire):
                d = denumire.upper()
                has_termo = 'TERMO+' in d or 'TERMO +' in d
                has_metal_inox = 'METAL' in d and 'INOX' in d
                
                # TERMO+ cu METAL + INOX = prioritate maxima
                if has_termo and has_metal_inox:
                    return 1
                # TERMO+ = prioritate 2
                if has_termo:
                    return 2
                # METAL + INOX = prioritate 3
                if has_metal_inox:
                    return 3
                return 10
            
            results.sort(key=lambda r: (get_puffer_priority(r['d']), -r.get('score', 0)))
        
        # REGULA SPECIALA 6: KIT TUR+RETUR+CAP TERMOSTATIC
        # Daca query contine TUR + RETUR + CAP/TERMOSTATIC -> kit Giacomini
        has_tur = 'TUR' in query_upper
        has_retur = 'RETUR' in query_upper or 'NET' in query_upper
        has_cap = 'CAP' in query_upper or 'TERMOSTA' in query_upper
        
        if has_tur and (has_retur or has_cap):
            # Verifica tipul de racord
            has_pex = 'PEX' in query_upper or 'PE-XA' in query_upper or 'PEXA' in query_upper or ' FE ' in query_upper or query_upper.endswith(' FE')
            has_ppr = 'PPR' in query_upper or ' FI ' in query_upper or query_upper.endswith(' FI')
            
            # Cauta produsele R470 in baza
            if has_pex and has_retur:
                # R470AX003 KIT TERMOSTATAT TUR+RETUR+TERMOSTAT 1/2X15*16 GIACOMINI (pentru PEX)
                for prod in PRODUCTS_DB:
                    if 'R470AX003' in prod['c'] or 'R470AX003' in prod['d']:
                        # Adauga la inceput daca nu e deja
                        kit_result = {'d': prod['d'], 'c': prod['c'], 'score': 99999, 'ratio': 1.0}
                        results = [kit_result] + [r for r in results if r['c'] != prod['c']]
                        break
            elif has_ppr and has_retur:
                # R470FX003 KIT TERMOSTATAT TUR+RETUR+TERMOSTAT 1/2" FI GIACOMINI (pentru PPR)
                for prod in PRODUCTS_DB:
                    if 'R470FX003' in prod['c'] or 'R470FX003' in prod['d']:
                        kit_result = {'d': prod['d'], 'c': prod['c'], 'score': 99999, 'ratio': 1.0}
                        results = [kit_result] + [r for r in results if r['c'] != prod['c']]
                        break
            elif has_retur:
                # Daca nu e specificat tipul, arata ambele kituri primele
                kit_results = [r for r in results if 'R470' in r['c'].upper() or ('KIT' in r['d'].upper() and 'TUR' in r['d'].upper())]
                other_results = [r for r in results if r not in kit_results]
                results = kit_results + other_results
        
        # REGULA SPECIALA 2: Vase expansiune - prioritate VR/VRV pentru boiler/centrala, VAV/VAO pentru hidrofor
        is_vas_query = 'VAS' in query_upper and ('LITRI' in query_upper or 'LIT' in query_upper or 'L ' in query_upper)
        if is_vas_query:
            is_boiler = 'BOILER' in query_upper or 'CENTRALA' in query_upper or 'INCALZIRE' in query_upper
            is_hidrofor = 'HIDROFOR' in query_upper or 'APA' in query_upper or 'RECE' in query_upper
            
            def get_vas_priority(cod, denumire):
                c = cod.upper()
                # VR = vas rosu fara suport (prioritar pentru boiler/centrala)
                if c.startswith('VR') and not c.startswith('VRV'):
                    return 1
                # VRV = vas rosu vertical cu suport
                if c.startswith('VRV'):
                    return 2
                # VAV = vas albastru vertical (prioritar pentru hidrofor)
                if c.startswith('VAV'):
                    return 3
                # VAO = vas albastru orizontal
                if c.startswith('VAO'):
                    return 4
                return 10
            
            if is_boiler or (not is_hidrofor):
                # Prioritate: VR > VRV > altele
                results.sort(key=lambda r: (get_vas_priority(r['c'], r['d']), -r.get('score', 0)))
            elif is_hidrofor:
                # Prioritate: VAV > VAO > altele
                def get_hidrofor_priority(cod):
                    c = cod.upper()
                    if c.startswith('VAV'): return 1
                    if c.startswith('VAO'): return 2
                    return 10
                results.sort(key=lambda r: (get_hidrofor_priority(r['c']), -r.get('score', 0)))
        
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
