
## Tokenizer 사용법

- soynlp_scores_path : `/media/scatter/scatterdisk/data/fasttext.soynlp_280K.256D/soynlp_scores.sol.100M.txt`

- 불러오는 법

~~~python
from soynlp.tokenizer import MaxScoreTokenizer

class SoyNLPTokenizer:
"""
Tokenize text using MaxScoreTokenizer of SoyNLP
"""
def __init__(self, soynlp_scores_path):
with open(soynlp_scores_path, "r") as f:
scores = [line.strip().split("\t") for line in f]
scores = {word: float(score) for word, score in scores}
self.tokenizer = MaxScoreTokenizer(scores=scores)

def tokenize(self, sentence):
tokenized_sentence = self.tokenizer.tokenize(sentence)
return tokenized_sentence
~~~

- 불러오신 후 `tokenize`쓰시면 됩니다.

## FastText Embedding 사용법
- 주의할 점
    - 모든 단어들이 자모단위로 저장되어 있습니다. (`안녕` -> `ㅇㅏㄴㄴㅕㅇ`)
    - 이를 쓰기 위해서는 `https://github.com/scatterlab/ml/blob/master/utils/util.py`의 `JamoProcessor`를 쓰셔야 합니다.

- pretrained_embedding_path : `/media/scatter/scatterdisk/data/fasttext.soynlp_280K.256D/fasttext.soynlp_280K.256D`

~~~python
from gensim.models import FastText

ft = FastText.load(pretrained_embedding_path)
~~~

- 불러오신 후 단어를 넣으실 때, 단어를 자모로 바꾸신 다음에 `most_similar()` 등을 사용해서 단어들을 비교할 수 있습니다.