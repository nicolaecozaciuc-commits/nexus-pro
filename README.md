# Nexus PRO - Server Edition v1.1

Aplicație pentru extragerea produselor din imagini folosind OpenAI GPT-4o.

## Structura fișierelor

```
nexus-pro/
├── app.py              # Server Flask
├── templates/
│   └── index.html      # Interfața web
├── requirements.txt    # Dependențe Python
├── start.sh            # Script pornire server
├── produse_nexus.csv   # Baza de date (de adăugat)
└── README.md
```

## Deploy pe Contabo (Ubuntu)

### 1. Clonează repository-ul
```bash
cd /root
git clone https://github.com/nicolaecozaciuc-commits/nexus-pro.git
cd nexus-pro
```

### 2. Configurare mediu virtual
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Adaugă baza de date
Urcă fișierul `produse_nexus.csv` sau `produse_nexus.xlsx` în folder.

### 4. Pornește serverul
```bash
chmod +x start.sh
./start.sh
```

Sau manual:
```bash
gunicorn -w 2 -b 0.0.0.0:8082 app:app --daemon
```

### 5. Accesează aplicația
```
http://IP_SERVER:8082
```

## Oprire server
```bash
pkill -f "gunicorn.*app:app"
```

## Port
Aplicația rulează pe **portul 8082**.
