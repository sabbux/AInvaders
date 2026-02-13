# Space Invaders AI: Imitation Learning vs. Reinforcement Learning

Questo progetto universitario ha l'obiettivo di sviluppare, addestrare e confrontare diverse architetture di Intelligenza Artificiale applicate al videogioco arcade **Space Invaders**.

Lo studio mette a confronto due paradigmi principali di apprendimento automatico:
1.  **Imitation Learning (Apprendimento Supervisionato):** Agenti che imparano clonando il comportamento di un "Maestro" esperto.
2.  **Reinforcement Learning (Apprendimento per Rinforzo):** Agenti che apprendono una strategia da zero tramite tentativi ed errori (trial-and-error).

## 🧠 Gli Agenti

Sono stati implementati e confrontati quattro agenti distinti:

| Agente | Tipo | Descrizione |
| :--- | :--- | :--- |
| **Heuristic (Maestro)** | Rule-Based | Un bot programmato con regole fisse "if-then". Funge da **Teacher**: non impara, ma gioca in modo eccellente per generare il dataset di "partite perfette" necessario agli agenti supervisionati. |
| **XGBoost** | Supervised | Un classificatore basato su alberi decisionali (Gradient Boosting). È addestrato per prevedere l'azione esatta del Maestro data una situazione di gioco. Estremamente veloce ed efficiente. |
| **MLP (Rete Neurale)** | Supervised | Un *Multi-Layer Perceptron* (Rete Neurale profonda). Cerca di generalizzare le regole del Maestro, catturando pattern più complessi rispetto agli alberi decisionali. |
| **PPO** | Reinforcement | Un agente basato su *Proximal Policy Optimization* (tramite Stable Baselines 3). Non vede mai i dati del Maestro; impara giocando milioni di frame e ricevendo "ricompense" (+punti) o "punizioni" (morte). |

## 📂 Struttura del Progetto

Il repository è organizzato in modo modulare per separare l'ambiente di gioco, il training e la valutazione.

```bash
.
├── agents/                 # Codice agenti e Modelli RL (PPO)
│   ├── heuristic_agent.py  # Il Maestro (genera anche il dataset)
│   ├── xgboost_agent.py    # Player per il modello XGBoost
│   ├── mlp_agent.py        # Player per il modello MLP
│   ├── ppo_space_invaders_stacked.zip # Modello PPO addestrato finale
│   ├── logs/               # Log di TensorBoard per il training RL
│   └── checkpoints/        # Salvataggi intermedi del modello PPO
├── environment/            # Motore di gioco modificato
│   ├── spaceinvaders.py    # Core del gioco (Pygame)
│   └── gym_env.py          # Wrapper Gymnasium per RL (PPO)
├── training/               # Script per l'addestramento dei modelli
│   ├── xgboost_train.py    # Training Classificatore XGBoost
│   ├── mlp_train.py        # Training Rete Neurale
│   └── rl_train.py         # Training PPO (Reinforcement Learning)
├── tournament/             # Script di valutazione finale
│   └── final_compare.py    # Confronta tutti gli agenti in un torneo
├── resources/              # Cartella output (creata automaticamente)
│   ├── dataset_heuristic.csv # Dati generati dal Maestro
│   ├── xgboost_brain.json    # Modello salvato XGBoost
│   ├── mlp_brain.pkl         # Modello salvato MLP
│   └── mlp_scaler.pkl        # Scaler per normalizzare i dati MLP
├── utils/                  # Funzioni di utilità condivise
│   └── game_utils.py       # Feature Engineering
├── images/                 # Asset grafici di gioco
├── sounds/                 # Effetti sonori
└── requirements.txt        # Dipendenze Python
```

## 🚀 Installazione

Per replicare gli esperimenti, è consigliato utilizzare un ambiente virtuale Python (la versione usata durante lo sviluppo di questo progetto è la versione 3.11.9)

1. **Clona la repository**
```bash
git clone [https://github.com/sabbux/AInvaders.git](https://github.com/sabbux/AInvaders.git)
cd AInvaders
```

2. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

## 🕹️ Come Replicare il Lavoro

Il flusso di lavoro si divide in tre fasi: Generazione Dati, Addestramento e Valutazione.

### Fase 1: Generazione del Dataset

Prima di addestrare i modelli supervisionati, bisogna creare il dataset. L'Agente Euristico gioca diverse partite e salva ogni decisione (Stato -> Azione) in un file CSV.

```bash
python agents/heuristic_agent.py
```

*Output:* Crea il file `resources/dataset_heuristic.csv`.

### Fase 2: Addestramento Supervisionato (Imitation Learning)

Addestriamo XGBoost e MLP utilizzando i dati generati dall'Euristico.

### Training XGBoost:
```bash
python training/xgboost_train.py
```

*Output:* Salva `resources/xgboost_brain.json`.

### Training MLP (Rete Neurale):
```bash
python training/mlp_train.py
```

*Output:* Salva `resources/mlp_brain.pkl` e `resources/mlp_scaler.pkl`.

### Fase 3: Addestramento Reinforcement Learning (PPO)

Addestriamo l'agente PPO da zero. Questo processo richiede più tempo poiché l'agente deve esplorare l'ambiente.
```bash
python training/rl_train.py
```

*Output:* Salva il modello e i checkpoint nella cartella `agents/`.

### Fase 4: Il Torneo Finale

Una volta che tutti i modelli sono pronti, esegui lo script di comparazione per vederli giocare uno dopo l'altro e confrontare i punteggi medi.
```bash
python tournament/final_compare.py
```

## 🛠️ Modifiche all'Ambiente di Gioco

L'ambiente di gioco originale è stato pesantemente modificato per scopi di ricerca:

- **Feature Extraction**: Il gioco non restituisce solo pixel, ma un vettore numerico di feature (posizione nemici, proiettili, distanze) utilizzabile dagli algoritmi ML.

- **Gymnasium Wrapper**: È stato creato un wrapper compatibile con lo standard `Gymnasium` per permettere l'integrazione con librerie di Reinforcement Learning come Stable Baselines 3.

- **Modalità Headless/Watch**: Il motore supporta l'esecuzione rapida senza rendering (per il training) e l'esecuzione visiva (per la valutazione).

## 🤝 Crediti e Riconoscimenti

Il motore grafico di base è basato sull'implementazione open-source di Space Invaders realizzata da [Lee Robinson](https://github.com/leerob/space-invaders).

Il codice originale è stato adattato per:

1. Esporre le variabili di stato interne per il Machine Learning.
2. Permettere il controllo esterno da parte di agenti AI.
3. Implementare logiche di anti-camping e difficoltà progressiva.

## 👥 Team

Il progetto è stato sviluppato da:
* [Francesco Sabetta](https://github.com/sabbux)
* [Monica Tiberini](https://github.com/tmmt9101)

---
Questa repository contiene la completa implementazione del progetto accademico del corso di `Fondamenti di Intelligenza Artificiale` di Unisa.
