import os
import re
import json
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURARE GLOBALĂ ---
# Baza de date va fi încărcată în memoria RAM a serverului
PRODUCTS_DB = []

def load_database():
    """
    Încarcă baza de date la pornirea serverului.
    Caută automat fișiere .csv sau .xlsx în folderul curent.
    """
    global PRODUCTS_DB
    print("🔄 Inițializez încărcarea bazei de date...")
    
    # Căutăm fișiere posibile
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx'))]
    file_path = files[0] if files else 'produse_nexus.csv' # Fallback
    
    if not os.path.exists(file_path):
        print(f"⚠️ ATENȚIE: Nu am găsit fișierul '{file_path}'. Urcă-l pe server!")
        return

    try:
        # Citire inteligentă (Excel sau CSV)
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, dtype=str)
        else:
            # Încercăm separatori comuni pentru CSV
            try:
                df = pd.read_csv(file_path, dtype=str, on_bad_lines='skip', engine='python')
            except:
                df = pd.read_csv(file_path, sep=';', dtype=str, on_bad_lines='skip', engine='python')

        # Normalizare coloane (elimină spații, face totul lowercase pentru detectie)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Detectare coloane cheie (Logică adaptată la fișierul tău)
        col_den = next((c for c in df.columns if 'denumire' in c), None)
        col_cod = next((c for c in df.columns if c == 'cod'), None)
        col_sel = next((c for c in df.columns if 'selectie' in c), None)
        
        if not col_den:
            # Fallback pe indici dacă nu găsim numele coloanelor
            # Presupunem structura: 0=Cod, 3=Denumire, 12=Selectie
            df = df.iloc[:, [0, 3, 12]]
            df.columns = ['cod_lung', 'denumire', 'cod_scurt']
        else:
            # Redenumim pentru consistență
            rename_map = {col_den: 'denumire'}
            if col_cod: rename_map[col_cod] = 'cod_lung'
            if col_sel: rename_map[col_sel] = 'cod_scurt'
            df = df.rename(columns=rename_map)

        # Umplem golurile și convertim la string
        df = df.fillna('')
        
        # Procesare finală pentru viteză
        # Creăm un câmp "search_text" care conține toate datele relevante
        clean_data = []
        for _, row in df.iterrows():
            den = str(row.get('denumire', '')).strip()
            if len(den) < 2 or den.lower() == 'denumire': continue
            
            c_lung = str(row.get('cod_lung', '')).strip()
            c_scurt = str(row.get('cod_scurt', '')).strip()
            
            # Codul final: Preferăm cel scurt
            final_code = c_scurt if c_scurt else c_lung
            
            clean_data.append({
                'd': den,
                'c': final_code,
                # String de căutare optimizat (lowercase)
                's': f"{den} {c_scurt} {c_lung}".lower()
            })
            
        PRODUCTS_DB = clean_data
        print(f"✅ SUCCES: {len(PRODUCTS_DB)} produse încărcate în memorie.")
        
    except Exception as e:
        print(f"❌ EROARE CRITICĂ la citirea bazei de date: {e}")

# Încărcăm baza la start
load_database()

# --- RUTE WEB ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """
    API endpoint pentru căutarea rapidă.
    Primește JSON: { "query": "robinet 1/2" }
    Returnează JSON: [ { "d": "Robinet...", "c": "123" }, ... ]
    """
    try:
        data = request.json
        query = data.get('query', '').lower().strip()
        
        if not query or len(query) < 2:
            return jsonify([])
            
        # Algoritm de căutare
        results = []
        parts = query.split()
        
        # Limităm căutarea la primele 50 rezultate pentru viteză
        count = 0
        limit = 30
        
        for prod in PRODUCTS_DB:
            # Verificăm dacă TOATE cuvintele din query există în produs
            # (Ex: "robinet 1/2" -> trebuie să aibă și "robinet" și "1/2")
            if all(part in prod['s'] for part in parts):
                results.append(prod)
                count += 1
                if count >= limit: break
        
        return jsonify(results)
        
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
