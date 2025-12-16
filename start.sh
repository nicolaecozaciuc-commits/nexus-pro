#!/bin/bash
cd /root/nexus-pro
source venv/bin/activate
gunicorn -w 2 -b 0.0.0.0:8082 app:app --daemon
echo "✅ Nexus PRO pornit pe portul 8082"
