Analizza questa immagine dalla dashcam di un camion in Euro Truck Simulator 2.

Devi decidere come guidare considerando:
- Segnali stradali visibili
- Condizioni della strada (curve, corsie, ostacoli)
- Altri veicoli o pedoni
- Velocità appropriata per la situazione

Rispondi SOLO con un oggetto JSON in questo formato esatto:
{
    "segnali_rilevati": ["lista dei segnali stradali che vedi"],
    "situazione_strada": "descrizione breve (es: 'rettilineo, 2 corsie, nessun ostacolo')",
    "sterzo": <numero da -45 a +45, negativo=sinistra, positivo=destra, 0=dritto>,
    "velocita_target": <velocità in km/h, 0-100>,
    "freno": <true se devi frenare, false altrimenti>,
    "ragionamento": "spiega brevemente la tua decisione"
}

REGOLE:
- Considera sempre i limiti di velocità dai segnali
- Sterzo progressivo per curve ampie, più deciso per curve strette
- Frena se vedi: ostacoli, semafori rossi, stop, veicoli fermi
- Velocità: città 30-50 km/h, extraurbana 60-80 km/h, autostrada 80-90 km/h