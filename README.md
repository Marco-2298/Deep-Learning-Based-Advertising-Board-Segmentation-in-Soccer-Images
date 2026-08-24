# Deep Learning-Based Advertising-Board Segmentation in Soccer Images

Progetto sviluppato per il corso di Deep Learning con l'obiettivo di individuare e segmentare automaticamente i pannelli pubblicitari presenti in immagini di partite di calcio.

La pipeline utilizzata è:

```text
Football frame
    ↓
RF-DETR
    ↓
Bounding boxes
    ↓
SAM 2
    ↓
Segmentation mask


RF-DETR viene utilizzato per la localizzazione dei pannelli pubblicitari, mentre le bounding box predette vengono fornite come prompt a SAM 2 per ottenere una segmentazione pixel-level.

Il progetto analizza inoltre i principali limiti della pipeline, con particolare attenzione alla qualità delle annotazioni, alla generalizzazione tra dataset differenti e alla propagazione degli errori tra detector e segmentatore.

Dataset

Sono stati utilizzati due dataset distinti.

Detection dataset

Il dataset di detection contiene 8.851 immagini.

Le sei categorie originali di sponsor sono state accorpate in una singola classe:

advertising_board

Dopo la conversione in formato COCO e il preprocessing, il dataset contiene 19.146 annotazioni.

Split	Images	Annotations
Train	6.196	13.349
Validation	1.328	2.858
Test	1.327	2.939

Lo split è stato generato utilizzando seed 42.

Segmentation dataset

Il dataset utilizzato per SAM 2 contiene:

1.620 immagini
1.620 ground-truth masks

Sono stati verificati pairing, dimensioni, presenza di maschere vuote e componenti connesse. Tutte le maschere contengono una singola componente foreground connessa.

I dataset non sono inclusi nel repository.

RF-DETR

Per la detection è stato utilizzato RF-DETR Nano.

Configurazione principale:

Resolution: 384
Epochs: 5
Batch size: 4
Gradient accumulation: 4
Effective batch size: 16

Il training è stato eseguito su Google Colab con GPU NVIDIA T4.

Test set originale
Metric	Value
mAP50:95	0.8797
mAP50	0.9883
mAP75	0.9786
mAR	0.9194
F1	0.9869
Precision	0.9871
Recall	0.9867
Revisione delle annotazioni

L'analisi qualitativa ha evidenziato che alcune regioni pubblicitarie visibili non erano presenti nelle annotazioni originali.

È stato quindi revisionato manualmente un sottoinsieme casuale di 100 immagini del test set.

Original annotations: 217
Revised annotations: 338
Added annotations:    121

Le annotazioni aumentano del 55,8%.

Valutando lo stesso checkpoint sulle annotazioni revisionate si ottiene:

Metric	Original GT	Revised GT
mAP50:95	0.8880	0.5804
mAP50	1.0000	0.7098
mAP75	0.9816	0.6419
mAR	0.9300	0.6781
F1	0.9977	0.7828
Precision	0.9954	0.9954
Recall	1.0000	0.6450

La precision rimane sostanzialmente invariata, mentre il recall diminuisce sensibilmente, evidenziando l'effetto del protocollo di annotazione sulla valutazione del detector.

SAM 2

Per la segmentazione è stato utilizzato il modello pre-trained:

facebook/sam2.1-hiera-small

senza fine-tuning.

Gli esperimenti sono stati eseguiti localmente su Apple Silicon tramite PyTorch MPS.

Sono state valutate due configurazioni.

Ground-truth bbox → SAM 2

Le bounding box vengono ricavate direttamente dalle maschere ground truth.

Metric	Value
Mean IoU	0.7834
Median IoU	0.8306
Micro IoU	0.7778
Mean Dice	0.8698
Median Dice	0.9075
Micro Dice	0.8750
RF-DETR → SAM 2

Le bounding box vengono prodotte automaticamente da RF-DETR con confidence threshold pari a 0.50.

Metric	Value
Mean IoU	0.5802
Micro IoU	0.5855
Mean Dice	0.6659
Micro Dice	0.7386

Il confronto mostra un calo delle prestazioni quando SAM 2 riceve le bounding box predette dal detector invece di prompt derivati dalla ground truth.

RF-DETR non produce alcuna detection in 227 frame su 1.620 alla soglia 0.50. Un'analisi diagnostica a soglie inferiori mostra che nella maggior parte di questi casi sono comunque presenti proposte a confidence più bassa.

Repository structure
football-adboard-segmentation/
├── configs/
│   └── experiment.yaml
├── data/
├── notebooks/
│   ├── 03_train_rfdetr_colab.ipynb
│   ├── 04_evaluate_rfdetr_colab.ipynb
│   ├── 05_review_test_subset_colab.ipynb
│   └── 06_sam2_segmentation_colab.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── utils.py
│   ├── validate_coco.py
│   └── visualize_coco.py
├── .gitignore
├── README.md
└── requirements.txt
Notebooks
03_train_rfdetr_colab.ipynb: training di RF-DETR Nano.
04_evaluate_rfdetr_colab.ipynb: valutazione e analisi qualitativa del detector.
05_review_test_subset_colab.ipynb: revisione manuale di 100 immagini e confronto tra ground truth originale e revisionata.
06_sam2_segmentation_colab.ipynb: esperimenti GT bbox → SAM 2 e RF-DETR → SAM 2.
Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Versione RF-DETR utilizzata:

rfdetr==1.9.1
Data preparation

Conversione e preparazione del detection dataset:

python -m src.prepare_data

Validazione della struttura COCO:

python -m src.validate_coco
Limitations

I principali limiti osservati riguardano:

incompletezza delle annotazioni;
differenza di dominio tra detection e segmentation dataset;
sponsor e contenuti grafici non presenti nel training set;
prospettiva e occlusioni;
dipendenza di SAM 2 dalla qualità della bounding box;
propagazione degli errori tra detection e segmentation.

Il progetto non include advertising replacement, inpainting o rimozione della rete della porta.

Conclusion

RF-DETR ottiene risultati elevati rispetto alle annotazioni originali, ma la revisione manuale mostra quanto la qualità della ground truth influenzi la valutazione.

SAM 2 produce buone segmentazioni quando riceve bounding box derivate dalla ground truth, mentre la pipeline end-to-end risente degli errori e della generalizzazione del detector.

Il progetto evidenzia quindi l'importanza della qualità delle annotazioni e dell'interazione tra i diversi stadi di una pipeline detector-segmenter.
