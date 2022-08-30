# crnn_transformer_mlm
## transformer_mlm

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


## split_agnostic

### Model Training
python train.py

### Recognition
python predict.py
