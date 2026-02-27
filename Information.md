## 📊 Machine Predictive Maintenance Classification (Kaggle)

Quelle: Kaggle  
Datentyp: Synthetischer Sensordatensatz zur Vorhersage von Maschinenfehlern  
Ziel: Entwicklung und Vergleich von Machine-Learning-Modellen zur Predictive Maintenance  

🔎 Überblick
- 10.000 Datensätze  
- Tabellarisches CSV-Format  
- Industrielle Maschinenzustände basierend auf Sensordaten  
- Geeignet für Klassifikationsaufgaben  

⚙️ Wichtige Eingangsmerkmale (Features)
- Lufttemperatur (K)  
- Prozesstemperatur (K)  
- Drehzahl (rpm)  
- Drehmoment (Nm)  
- Werkzeugverschleiß (min)  
- Produkttyp (L / M / H)  

➡️ Diese Variablen dienen als Grundlage zur Fehlerprognose.  

🎯 Zielvariablen
- target → 0 = kein Fehler | 1 = Fehler  
- failure_type → Art des Fehlers (nur bei Fehler = 1 relevant)  

⚠️ Hinweis: failure_type darf nicht als Eingabevariable verwendet werden (Data Leakage).  

🧠 Typische Anwendungen
- Binäre Klassifikation (Fehler ja/nein)  
- Multiklassen-Klassifikation (Fehlertyp)  
- Vergleich von ML-Modellen (z. B. Random Forest, Logistic Regression, XGBoost)  

📌 Besonderheit
Der Datensatz ist synthetisch erzeugt, bildet jedoch realistische industrielle Wartungsszenarien ab und eignet sich ideal für:
- Studienprojekte  
- ML-Übungen  
- Modellvergleiche  
- Feature-Engineering  
