# Deep Learning-Based Advertising-Board Segmentation in Soccer Images

Progetto sviluppato per il corso di Deep Learning con l'obiettivo di individuare e segmentare automaticamente i pannelli pubblicitari presenti in immagini di partite di calcio.

La pipeline utilizzata è composta da due modelli:

**RF-DETR → bounding box → SAM 2 → segmentation mask**

RF-DETR viene utilizzato per localizzare i pannelli pubblicitari, mentre le bounding box prodotte dal detector vengono utilizzate come prompt per SAM 2, che genera la corrispondente maschera pixel-level.

Il progetto è stato utilizzato anche per analizzare i limiti della pipeline, in particolare rispetto alla qualità delle annotazioni, alla generalizzazione del detector su immagini provenienti da dataset differenti e alla propagazione degli errori tra detection e segmentation.

## Pipeline

Il flusso principale è:

```text
Football frame
      |
      v
   RF-DETR
      |
      v
Bounding boxes
      |
      v
    SAM 2
      |
      v
Segmentation mask


Sono state valutate separatamente due configurazioni:

GT bbox → SAM 2

Le bounding box vengono ricavate direttamente dalle maschere ground truth.
Questo esperimento permette di valutare SAM 2 utilizzando un prompt geometrico ideale.

RF-DETR → SAM 2

Le bounding box vengono generate automaticamente da RF-DETR e fornite a SAM 2.
Questa configurazione rappresenta la pipeline end-to-end effettiva.

È stata inoltre utilizzata una variante GT-aligned esclusivamente come analisi diagnostica.

Dataset

Sono stati utilizzati due dataset differenti.

Detection dataset

Dataset composto da 8.851 immagini di partite di calcio con annotazioni relative ai pannelli pubblicitari.

Le annotazioni originali distinguono sei categorie di sponsor:

Heineken
Mastercard
Gazprom
PlayStation
Nissan
Pepsi

Per il progetto queste categorie sono state accorpate in una singola classe:

advertising_board

Dopo la conversione delle annotazioni nel formato COCO e il filtraggio delle regioni non valide, il dataset finale contiene 19.146 bounding box.

Lo split utilizzato è:

Split	Images	Annotations
Train	6.196	13.349
Validation	1.328	2.858
Test	1.327	2.939

Lo split è stato generato utilizzando seed 42.

Segmentation dataset

Il secondo dataset contiene:

1.620 immagini
1.620 ground-truth masks

Ogni immagine è associata a una maschera binaria pixel-level.

Prima degli esperimenti sono stati verificati:

pairing tra immagini e maschere;
compatibilità delle dimensioni;
assenza di maschere vuote;
valori presenti nelle maschere;
numero di componenti connesse.

Tutte le maschere analizzate contengono una singola componente foreground connessa.

I dataset non sono inclusi nel repository.

RF-DETR

Per la fase di detection è stato utilizzato RF-DETR Nano.

Configurazione principale:

Model: RF-DETR Nano
Resolution: 384
Epochs: 5
Batch size: 4
Gradient accumulation: 4
Effective batch size: 16

Il training è stato eseguito su Google Colab utilizzando una GPU NVIDIA T4.

È stato utilizzato il checkpoint con le migliori prestazioni di validation.

Risultati sul test set originale
Metric	Value
mAP50:95	0.8797
mAP50	0.9883
mAP75	0.9786
mAR	0.9194
F1	0.9869
Precision	0.9871
Recall	0.9867

Le metriche iniziali risultano molto elevate. Tuttavia, durante l'analisi qualitativa è emerso che alcune regioni pubblicitarie visibili nelle immagini non erano presenti nelle annotazioni originali.

Review delle annotazioni

Per analizzare il problema è stato estratto casualmente, con seed 42, un sottoinsieme di 100 immagini dal test set.

Le annotazioni sono state revisionate manualmente mantenendo invariata la definizione della classe advertising_board.

Il numero di bounding box è passato da:

Original annotations: 217
Revised annotations: 338
Added annotations:    121

corrispondenti a un aumento del 55,8% rispetto alle annotazioni originali.

Lo stesso checkpoint RF-DETR è stato quindi valutato sulle stesse 100 immagini utilizzando prima le annotazioni originali e successivamente quelle revisionate.

Original ground truth
Metric	Value
mAP50:95	0.8880
mAP50	1.0000
mAP75	0.9816
mAR	0.9300
F1	0.9977
Precision	0.9954
Recall	1.0000
Revised ground truth
Metric	Value
mAP50:95	0.5804
mAP50	0.7098
mAP75	0.6419
mAR	0.6781
F1	0.7828
Precision	0.9954
Recall	0.6450

La precision rimane praticamente invariata, mentre il recall diminuisce sensibilmente.

Questo risultato indica che il detector tende a localizzare correttamente le regioni compatibili con il protocollo di annotazione utilizzato durante il training, ma non identifica necessariamente tutte le regioni pubblicitarie visibili secondo una definizione più completa del problema.

SAM 2

Per la fase di segmentazione è stato utilizzato:

facebook/sam2.1-hiera-small

Il modello è stato utilizzato pre-trained, senza fine-tuning.

Gli esperimenti di segmentazione sono stati eseguiti localmente su Apple Silicon tramite backend PyTorch MPS.

Le prestazioni sono state valutate utilizzando:

Intersection over Union (IoU)
Dice coefficient
Experiment A: Ground-truth boxes → SAM 2

Nel primo esperimento la bounding box viene ricavata direttamente dalla ground-truth mask e utilizzata come prompt per SAM 2.

Risultati sulle 1.620 immagini:

Metric	Value
Mean IoU	0.7834
Median IoU	0.8306
Micro IoU	0.7778
Mean Dice	0.8698
Median Dice	0.9075
Micro Dice	0.8750

Questa configurazione permette di osservare il comportamento di SAM 2 quando il prompt fornito è direttamente derivato dalla regione annotata.

Experiment B: RF-DETR → SAM 2

Nel secondo esperimento la bounding box viene prodotta automaticamente da RF-DETR.

La soglia di confidence utilizzata è:

0.50

Risultati sulle 1.620 immagini:

Metric	Value
Mean IoU	0.5802
Micro IoU	0.5855
Mean Dice	0.6659
Micro Dice	0.7386

Il passaggio da bounding box ground truth a bounding box predette determina quindi un calo significativo delle prestazioni.

Questo evidenzia la propagazione dell'errore nella pipeline: la qualità della segmentazione finale dipende non soltanto da SAM 2, ma anche dalla qualità e dalla disponibilità dei prompt prodotti dal detector.

Detection analysis sul segmentation dataset

Utilizzando una threshold pari a 0.50, RF-DETR non produce alcuna detection in 227 frame su 1.620, circa il 14% del dataset.

Questi frame sono stati analizzati nuovamente riducendo, a solo scopo diagnostico, la soglia fino a 0.05.

Nella maggior parte dei casi il detector produce comunque delle proposte, ma con confidence inferiore alla soglia utilizzata nell'esperimento principale.

Solo tre frame non presentano alcuna detection nemmeno con threshold 0.05.

La soglia ufficiale della pipeline è comunque rimasta pari a 0.50; l'analisi a threshold inferiori è stata utilizzata solamente per studiare il comportamento del detector nei casi di fallimento.

Analisi qualitativa

Oltre alle metriche aggregate sono stati analizzati alcuni casi rappresentativi della pipeline.

Sono stati selezionati:

un caso con IoU elevata;
un caso vicino alla mediana;
un caso con IoU molto bassa ma con almeno una detection RF-DETR.

L'analisi mostra che una confidence elevata del detector non implica necessariamente una segmentazione finale corretta.

In alcuni casi la bounding box prodotta dal detector non rappresenta in modo sufficientemente coerente la regione descritta dalla ground truth oppure il prompt porta SAM 2 a produrre una maschera non allineata con il target.

Repository structure
football-adboard-segmentation/
├── configs/
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 03_train_rfdetr_colab.ipynb
│   ├── 04_evaluate_rfdetr_colab.ipynb
│   ├── 05_review_test_subset_colab.ipynb
│   └── 06_sam2_segmentation_colab.ipynb
│
├── outputs/
│   ├── figures/
│   ├── reviewed_test_subset_100/
│   └── sam2/
│
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── utils.py
│   ├── validate_coco.py
│   └── visualize_coco.py
│
├── .gitignore
├── README.md
└── requirements.txt
Main notebooks
03_train_rfdetr_colab.ipynb

Training di RF-DETR Nano sul detection dataset.

Contiene:

verifica del dataset COCO;
preparazione dei dati su Colab;
training;
resume del training;
analisi delle validation metrics;
selezione del checkpoint finale.
04_evaluate_rfdetr_colab.ipynb

Valutazione del detector sul test set.

Contiene:

metriche COCO;
visualizzazione delle predizioni;
confronto qualitativo con la ground truth;
review manuale della completezza delle annotazioni.
05_review_test_subset_colab.ipynb

Revisione manuale di un sottoinsieme di 100 immagini del test set.

Il notebook confronta le prestazioni dello stesso checkpoint RF-DETR utilizzando la ground truth originale e quella revisionata.

06_sam2_segmentation_colab.ipynb

Esperimenti di segmentazione con SAM 2.

Contiene:

analisi del segmentation dataset;
GT bbox → SAM 2;
RF-DETR → SAM 2;
confronto quantitativo;
analisi dei frame senza detection;
analisi qualitativa dei casi migliori, mediani e peggiori.
Setup

Creare un ambiente Python e installare le dipendenze:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Per RF-DETR è stata utilizzata la versione:

rfdetr==1.9.1

SAM 2 viene utilizzato attraverso il repository ufficiale Meta e il modello pre-trained facebook/sam2.1-hiera-small.

Data preparation

La conversione del detection dataset nel formato COCO può essere eseguita tramite:

python -m src.prepare_data

La struttura generata può essere verificata tramite:

python -m src.validate_coco

La preparazione del dataset non deve essere rieseguita se si vogliono riprodurre esattamente gli stessi split utilizzati negli esperimenti.

Limitations

I principali limiti osservati sono legati a:

incompletezza delle annotazioni del detection dataset;
differenza tra i domini dei dataset di detection e segmentation;
nuovi sponsor e differenti contenuti grafici;
forte prospettiva dei pannelli;
occlusioni dovute a giocatori, porte e reti;
dipendenza di SAM 2 dalla qualità della bounding box utilizzata come prompt;
propagazione degli errori tra detector e segmentatore.

Il progetto non affronta la sostituzione dei contenuti pubblicitari, l'inpainting o la rimozione della rete della porta.

Conclusions

RF-DETR raggiunge prestazioni elevate quando viene valutato rispetto al protocollo di annotazione utilizzato durante il training. La revisione manuale del test set mostra però che le metriche originali possono sovrastimare la capacità del modello di individuare tutte le regioni pubblicitarie visibili.

SAM 2 produce segmentazioni di buona qualità quando viene utilizzato con bounding box derivate dalla ground truth. Le prestazioni diminuiscono nella pipeline end-to-end, dove i prompt dipendono direttamente dalle predizioni di RF-DETR.

I risultati mostrano quindi che, in una pipeline detector-segmenter, la qualità finale non dipende esclusivamente dai singoli modelli, ma anche dalla coerenza delle annotazioni, dalla generalizzazione tra domini differenti e dalla propagazione degli errori tra i diversi stadi.
