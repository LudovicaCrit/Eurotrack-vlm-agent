"""
VLM Agent per Guida Autonoma - Euro Truck Simulator 2
Progetto: Confronto VLM vs Reinforcement Learning
"""

import cv2
import json
import os
from dotenv import load_dotenv
import time
import re
import google.generativeai as genai
from PIL import Image

# Configurazione
load_dotenv()
import prompts

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
VIDEO_PATH = "ets2-clip.mp4"
FRAME_OGNI_SECONDI = 2
OUTPUT_DIR = "output"

if not GEMINI_API_KEY:
    print("ERRORE: GEMINI_API_KEY non trovata nel file .env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-flash-latest')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Funzioni principali


def frame_to_pil(frame):
    """Converte frame OpenCV in formato PIL per Gemini"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def analizza_frame(frame):
    """Analizza frame con VLM e restituisce comandi di guida"""
    pil_image = frame_to_pil(frame)
    response = model.generate_content([prompts.DRIVING_ANALYSIS_PROMPT, pil_image])
    text = response.text
    
    # Estrai JSON dalla risposta
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        return json.loads(text)


# Opzione 1: Test su singolo frame


def test_singolo_frame():
    print("\n" + "="*70)
    print("TEST SU SINGOLO FRAME")
    print("="*70)
    
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        print(f"ERRORE: Impossibile aprire {VIDEO_PATH}")
        return False
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    durata = total_frames / fps
    
    print(f"\nVideo caricato: {durata:.1f}s, {fps:.0f} FPS")
    
    frame_pos = total_frames // 2
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = video.read()
    video.release()
    
    if not ret:
        print("ERRORE: Impossibile leggere frame")
        return False
    
    test_frame_path = os.path.join(OUTPUT_DIR, "test_frame.jpg")
    cv2.imwrite(test_frame_path, frame)
    print(f"Frame salvato: {test_frame_path}")
    
    print("\nInvio frame al modello (attendere 3-5 secondi)...")
    
    try:
        risultato = analizza_frame(frame)
        
        print("\nRISULTATO:")
        print("-" * 70)
        print(json.dumps(risultato, indent=2, ensure_ascii=False))
        print("-" * 70)
        
        print("\nInterpretazione:")
        print(f"  Segnali: {', '.join(risultato['segnali_rilevati']) if risultato['segnali_rilevati'] else 'Nessuno'}")
        print(f"  Situazione: {risultato['situazione_strada']}")
        print(f"  Sterzo: {risultato['sterzo']:+.0f} gradi")
        print(f"  Velocita: {risultato['velocita_target']} km/h")
        print(f"  Freno: {'Si' if risultato['freno'] else 'No'}")
        print(f"  Ragionamento: {risultato['ragionamento']}")
        
        return True
        
    except Exception as e:
        print(f"\nERRORE: {e}")
        return False


# Opzione 2: Processing video completo


def processa_video_completo():
    print("\n" + "="*70)
    print("PROCESSING VIDEO COMPLETO")
    print("="*70)
    
    video = cv2.VideoCapture(VIDEO_PATH)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    durata = total_frames / fps
    
    frame_interval = int(fps * FRAME_OGNI_SECONDI)
    frames_da_analizzare = int(durata / FRAME_OGNI_SECONDI)
    
    print(f"\nParametri:")
    print(f"  Video: {VIDEO_PATH} ({durata:.1f}s)")
    print(f"  Frame da analizzare: {frames_da_analizzare}")
    print(f"  Frequenza: 1 frame ogni {FRAME_OGNI_SECONDI}s")
    print(f"  Tempo stimato: {frames_da_analizzare * 4 / 60:.1f} minuti")
    
    risposta = input("\nAvviare processing? (s/n): ").strip().lower()
    if risposta != 's':
        print("Operazione annullata.")
        return
    
    risultati = []
    frame_count = 0
    analizzati = 0
    errori = 0
    start_time = time.time()
    
    print("\nProcessing in corso...\n")
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            print(f"  {timestamp:6.1f}s | Frame {analizzati+1}/{frames_da_analizzare} | ", end='', flush=True)
            
            try:
                comando = analizza_frame(frame)
                risultati.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'comando': comando
                })
                print(f"OK - Sterzo:{comando['sterzo']:+4.0f} Vel:{comando['velocita_target']:3.0f}km/h")
                analizzati += 1
                
            except Exception as e:
                print(f"ERRORE: {str(e)[:50]}")
                risultati.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'errore': str(e)
                })
                errori += 1
        
        frame_count += 1
    
    video.release()
    tempo_totale = time.time() - start_time
    
    output_data = {
        'metadata': {
            'video': VIDEO_PATH,
            'modello': 'gemini-flash-latest',
            'durata_secondi': durata,
            'frame_analizzati': analizzati,
            'frame_errori': errori,
            'tempo_processing_secondi': tempo_totale
        },
        'risultati': risultati
    }
    
    output_path = os.path.join(OUTPUT_DIR, "risultati.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETATO")
    print(f"{'='*70}")
    print(f"  Frame analizzati: {analizzati}")
    print(f"  Frame con errori: {errori}")
    print(f"  Tempo totale: {tempo_totale/60:.1f} minuti")
    print(f"  Tempo medio per frame: {tempo_totale/analizzati:.1f}s")
    print(f"  Risultati salvati: {output_path}")
    print(f"\nProssimo passo: Opzione 3 per generare grafici e video demo")


# Opzione 3: Visualizzazione risultati


def visualizza_risultati():
    print("\n" + "="*70)
    print("GENERAZIONE GRAFICI E VIDEO")
    print("="*70)
    
    results_path = os.path.join(OUTPUT_DIR, "risultati.json")
    if not os.path.exists(results_path):
        print(f"\nERRORE: File {results_path} non trovato")
        print("Eseguire prima l'Opzione 2 (Processa video completo)")
        return
    
    print("\nCaricamento risultati...")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    risultati = [r for r in data['risultati'] if 'comando' in r]
    print(f"Caricati {len(risultati)} frame validi")
    
    if len(risultati) == 0:
        print("ERRORE: Nessun frame valido trovato")
        return
    
    # Estrazione dati
    timestamps = [r['timestamp'] for r in risultati]
    sterzo = [r['comando']['sterzo'] for r in risultati]
    velocita = [r['comando']['velocita_target'] for r in risultati]
    freno = [r['comando']['freno'] for r in risultati]
    freno_num = [1 if f else 0 for f in freno]
    
    # Generazione grafici
    print("\nGenerazione grafici...")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    ax1.plot(timestamps, sterzo, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(timestamps, sterzo, alpha=0.3)
    ax1.set_ylabel('Sterzo (gradi)', fontsize=12, fontweight='bold')
    ax1.set_title('VLM Agent - Analisi Comandi di Guida', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(timestamps, velocita, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.fill_between(timestamps, velocita, alpha=0.3, color='green')
    ax2.set_ylabel('Velocita (km/h)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3.fill_between(timestamps, freno_num, alpha=0.5, color='red', step='post')
    ax3.set_ylabel('Freno', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Tempo (secondi)', fontsize=12, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Si'])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    grafici_path = os.path.join(OUTPUT_DIR, "grafici.png")
    plt.savefig(grafici_path, dpi=300, bbox_inches='tight')
    print(f"  Salvato: {grafici_path}")
    
    # Statistiche
    print("\nStatistiche:")
    print(f"  Sterzo medio: {np.mean(np.abs(sterzo)):.1f} gradi")
    print(f"  Velocita media: {np.mean(velocita):.1f} km/h")
    print(f"  Numero frenate: {sum(freno_num)}")
    
    # Generazione video annotato
    print("\nGenerazione video annotato (richiede 1-2 minuti)...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = os.path.join(OUTPUT_DIR, "video_demo.mp4")
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
    
    risultati_dict = {r['frame']: r['comando'] for r in risultati}
    frame_count = 0
    ultimo_comando = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in risultati_dict:
            ultimo_comando = risultati_dict[frame_count]
        
        if ultimo_comando:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (520, 190), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            y_pos = 45
            cv2.putText(frame, f"Sterzo: {ultimo_comando['sterzo']:+.0f} gradi", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos += 40
            cv2.putText(frame, f"Velocita: {ultimo_comando['velocita_target']:.0f} km/h", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos += 40
            freno_text = "FRENO: SI" if ultimo_comando['freno'] else "Freno: no"
            freno_color = (0, 0, 255) if ultimo_comando['freno'] else (0, 255, 0)
            cv2.putText(frame, freno_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, freno_color, 2)
            
            if ultimo_comando['segnali_rilevati']:
                y_pos += 40
                segnali = ", ".join(ultimo_comando['segnali_rilevati'][:2])
                if len(segnali) > 50:
                    segnali = segnali[:47] + "..."
                cv2.putText(frame, f"Segnali: {segnali}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            percent = (frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100
            print(f"  Progresso: {percent:.0f}%", end='\r')
    
    cap.release()
    out.release()
    print(f"  Salvato: {video_output}       ")
    
    print(f"\n{'='*70}")
    print("VISUALIZZAZIONE COMPLETATA")
    print(f"{'='*70}")
    print(f"\nFile generati in '{OUTPUT_DIR}/':")
    print(f"  - grafici.png (grafici analisi)")
    print(f"  - video_demo.mp4 (video con overlay)")
    print(f"  - risultati.json (dati completi)")


# Menu principale


def main():
    print("\n" + "="*70)
    print("VLM AGENT - GUIDA AUTONOMA")
    print("Progetto: Confronto Vision-Language Models vs Reinforcement Learning")
    print("="*70)
    
    while True:
        print("\nMENU:")
        print("  1. Test su singolo frame")
        print("  2. Processa video completo")
        print("  3. Genera grafici e video demo")
        print("  4. Esci")
        
        scelta = input("\nSelezione: ").strip()
        
        if scelta == "1":
            test_singolo_frame()
        elif scelta == "2":
            processa_video_completo()
        elif scelta == "3":
            visualizza_risultati()
        elif scelta == "4":
            print("\nProgramma terminato.")
            break
        else:
            print("\nOpzione non valida. Selezionare un numero da 1 a 4.")


if __name__ == "__main__":
    main()