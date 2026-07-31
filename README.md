Segmentazione dei pannelli pubblicitari nel calcio

Progetto per il corso di Deep Learning: rilevamento e segmentazione dei pannelli pubblicitari nelle riprese calcistiche.

Obiettivo

Il progetto valuta una pipeline composta da due fasi:

Frame calcistico -> detector RF-DETR -> bounding box -> SAM 2 -> maschera del pannello

RF-DETR localizza ogni pannello pubblicitario. La bounding box ottenuta viene poi usata come prompt per SAM 2, che produce una maschera a livello di pixel. La valutazione distingue le prestazioni del detector dalla qualità della segmentazione con bounding box annotata o predetta.

Il progetto non implementa la sostituzione della pubblicità, l'inpainting o la rimozione della rete della porta: sono possibili estensioni future.

Struttura del repository

football-adboard-segmentation/
├── configs/                # Parametri degli esperimenti
├── data/                   # Dataset locali (non tracciati da Git)
├── notebooks/              # Notebook per esplorazione e visualizzazione
├── output/                # Predizioni, figure e metriche (non tracciati)
├── src/                    # Script Python riproducibili
├── README.md
├── requirements.txt
└── .gitignore

Dati

I dataset e i checkpoint preaddestrati sono esclusi dal repository per le loro dimensioni e licenze. Dopo aver scaricato i dataset, inseriscili in data/ e documenta fonte e licenza in data/README.md.

I dati previsti sono:

un dataset per il rilevamento con immagini calcistiche e bounding box annotate dei pannelli pubblicitari;

un dataset per la segmentazione con immagini e maschere binarie dei pannelli.

src/prepare_data.py convertirà le annotazioni originali nel formato richiesto dagli script di addestramento e valutazione.

Installazione

Si consiglia Python 3.10 o 3.11. I comandi seguenti usano un ambiente virtuale chiamato .venv.

macOS / Linux

cd football-adboard-segmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Windows PowerShell

cd football-adboard-segmentation
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Se PowerShell blocca lo script di attivazione, esegui una volta:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Esecuzione del progetto

Attiva prima l'ambiente virtuale, poi esegui gli script dalla cartella principale del repository:

python src/prepare_data.py --config configs/experiment.yaml
python src/train_detector.py --config configs/experiment.yaml
python src/evaluate_detector.py --config configs/experiment.yaml
python src/segment_with_sam.py --config configs/experiment.yaml
python src/evaluate_masks.py --config configs/experiment.yaml

Le opzioni esatte da riga di comando verranno aggiunte insieme agli script. Le impostazioni degli esperimenti devono essere memorizzate in configs/experiment.yaml, senza valori scritti direttamente nel codice.

Per lavorare con i notebook:

jupyter lab

In VS Code seleziona l'interprete .venv da Python: Select Interpreter; anche i notebook devono usare lo stesso kernel .venv.

Valutazione

Componente

Metriche

Detector RF-DETR

mAP@50 e mAP@50:95

SAM 2 con bounding box annotata

IoU e Dice

Pipeline completa: bounding box RF-DETR + SAM 2

IoU e Dice

Il confronto tra le ultime due righe permette di capire se un errore di segmentazione dipende principalmente dal detector o dal segmentatore.

Output

outputs/metrics/: file JSON/CSV delle metriche e curve di addestramento;

outputs/predictions/: bounding box e maschere predette;

outputs/figures/: immagini per relazione e slide, compresi casi riusciti e casi difficili.

Riproducibilità

Mantieni seed casuale, versione del modello, dimensione delle immagini, soglia di confidenza, epoche e suddivisione del dataset in configs/experiment.yaml.

Non caricare su Git dati grezzi, checkpoint o output generati molto pesanti.

Salva le versioni di Python e dei pacchetti usate nell'esecuzione finale (pip freeze > outputs/metrics/environment.txt).

Limiti attesi

I principali casi difficili sono distorsione prospettica, giocatori o arbitri che coprono il pannello, rete della porta, motion blur, ombre/riflessi e pannelli molto chiari o molto scuri.

Riferimenti

RF-DETR

Segment Anything 2