# text-matching-tensorflow

Tensorflow implementation of text matching models

## Version
tensorflow == 1.9.0 

## Arguments 
### Path Arguments
Parameter | Description  | Default
------------------------------------- | :------: | :------:
name            | 저장하거나 불러올 이름     | 
config         | config 경로가 있을 경우 그 이름 | ""
train_dir        | 학습 데이터가 있는 폴더나 학습 데이터 text 파일 | 
val_dir | 평가 데이터가 있는 폴더나 학습 데이터 text 파일 | 
checkpoint_dir | 모델을 저장할 폴더. 그 아래에 `name` 폴더 안에 저장됨 |

### Preprocess Arguments
Parameter | Description  | Default
------------------------------------- | :------: | :------:
sent_piece_model | SentPieceTokenizer 모델 저장 경로 | 
soynlp_scores         | MaxScoreTokenizer를 위한 Word score 저장 경로 | 
tokenizer        | 어떤 tokenizer를 선택할 것인지 | DummyTokenizer
normalizer | 어떤 normalizer를 선택할 것인지 | DummyNormalizer
pretrained_embed_dir | FastText 모델이 저장되어 있는 경로 | 
vocab_size | FastText 모델의 대략적인 vocab size | 
vocab_list | FastText 모델의 vocab들 리스트를 저장한 경로 |
min_length | 문장의 최소 토큰 수 | 1
max_length | 문장의 최대 토큰 수 | 30

### Training Arguments
Parameter | Description  | Default
------------------------------------- | :------: | :------:
batch_size | 배치 사이즈 | 512
num_epochs | 전체 epoch 수 | 10
evaluate_every | 몇 **step**마다 평가할 지 | 20000
save_every | 몇 **step**마다 모델을 저장할 지 | 20000
shuffle | 데이터를 섞으면서 넣을지 | True
learning_rate | AdamOptimizer에 넣을 learning rate | 1e-3

### Model Arguments
Parameter | Description  | Default
------------------------------------- | :------: | :------:
model | 학습할 모델 이름 | DualEncoderLSTM
embed_dim | pretrained embedding의 dimension | 256
lstm_dim | rnn의 hidden size | 512

### Sampling Arguments
Parameter | Description  | Default
------------------------------------- | :------: | :------:
num_negative_samples | negative sample 수 | 4
negative_sampling | negative sampling 방법 | random

### Regularization Arguments
Parameter | Description  | Default
------------------------------------- | :------: | :------:
embed_dropout_keep_prob | embedding dropout | 1
lstm_dropout_keep_prob | lstm dropout | 0.8
dense_dropout_keep_prob | dense dropout | 1

## Author
Sungnam Park, Scatterlab (sungnam1108@naver.com)
