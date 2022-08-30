# crnn_transformer_mlm

## 数据集PrIMus
#### PrIMus dataset can be downloaded from <https://grfia.dlsi.ua.es/primus/>, cite from paper [End-to-End Neural Optical Music Recognition of Monophonic Scores], and saved in this file.

## transformer_mlm file

### Model Training

#### Semantic
python train.py -encoding semnatic -save_model ./trained_semantic_model

#### Agnostic
python train.py -encoding agnostic -save_model ./trained_agnostic_model

### Recognition

#### Semantic
python predict.py -encoding semnatic

#### Agnostic
python predict.py -encoding agnostic


## split_agnostic file

### Model Training
python train.py

### Recognition
python predict.py
