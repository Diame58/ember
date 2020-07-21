<img src="resources/logo.png" align="right" width="250px" height="250px">

# Endgame Malware BEnchmark for Research

EMBER 데이터셋은 PE파일(features) 모음으로 악성코드 연구자를 위해 벤치마크 데이터셋을 제공합니다. EMBER2017 데이터셋은 2017년까지 스캔된 110만개 PE 파일(features)이며, EMBER2018 데이터셋은 2018년까지 1백만개 PE파일(features)입니다. 이 저장소를 통해 벤치마크 모델을 쉽게 재현하고, 제공된 데이터셋을 확장해 벤치마크 모델을 새롭게 PE 파일(hwp, pdf, jpg 등)로 분류할 수 있습니다.

이 링크는 데이터셋 관련 자세한 논문입니다: [https://arxiv.org/abs/1804.04637](https://arxiv.org/abs/1804.04637)

## 기능

[LIEF](https://lief.quarkslab.com/) 프로젝트는 EMBER 데이터셋 PE파일(features)을 추출하는데 사용됩니다. LIEF 프로젝트는 EMBER 데이터셋 PE파일 features 을 추출하는데 사용됩니다. Raw 파일(features)은 JSON 포맷으로 추출된다. 그리고 공개적으로 이용 가능한 데이터셋으로 포함됩니다. Vectorized(features)은 Raw(features)으로부터 생성됩니다. CSV, dataframe 등은 바이너리 포맷으로 저장됩니다. 이 저장소를 사용하면 PE파일을 Raw(features)와 Vectorized(features)로 쉽게 생성할 수 있습니다. 연구자는 자체 기능을 구현하거나 기존 기능과 다르게 벡터화 할 수 있습니다.

feature calculation version 입니다. Feature version 1 은 LIEF library version 0.8.3 으로 계산됩니다. Feature version 2 는 additional data directory feature, updated ordinal import processing 포함되고 LIEF library version 0.9.0 으로 계산됩니다. 윈도우/리눅스에서 LIEF version 0.10.1 으로 두 환경 모두 Feature version 2 작동됨을 확인했습니다.

## 년도별

첫 번째, EMBER 데이터셋은 2017년까지 수집된 샘플을 계산해 version 1 features 구성했습니다.  두 번째, EMBER 데이터셋은 2018년까지 수집된 샘플을 계산해 version 1 features 구성했습니다. EMBER2017 version 2 는 EMBER2017 version 1 까지 포함했습니다. 2017년부터 2018년까지 데이터를 조합해 사용하면 PE파일(features)의 장기적인 유형 진화에 대해 연구 가능합니다. 하지만, 2017년과 2018년 샘플을 선택할 때 기준을 잘 확인해야 합니다. 특히, 2018년 샘플을 선택하면 머신러닝 알고리즘으로 test sets, resultant training 하기 어려워집니다. 이 부분은 몇 년간 연구했지만 양해바라며 유의바랍니다.

## 다운로드

여기서 데이터를 다운로드 하십시오:

| Year | Feature Version | Filename                     | URL                                                                                                                              | sha256                                                             |
|------|-----------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 2017 | 1               | ember_dataset.tar.bz2        | [https://pubdata.endgame.com/ember/ember_dataset.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset.tar.bz2)               | `a5603de2f34f02ab6e21df7a0f97ec4ac84ddc65caee33fb610093dd6f9e1df9` |
| 2017 | 2               | ember_dataset_2017_2.tar.bz2 | [https://pubdata.endgame.com/ember/ember_dataset_2017_2.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset_2017_2.tar.bz2) | `60142493c44c11bc3fef292b216a293841283d86ff58384b5dc2d88194c87a6d` |
| 2018 | 2               | ember_dataset_2018_2.tar.bz2 | [https://pubdata.endgame.com/ember/ember_dataset_2018_2.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset_2018_2.tar.bz2) | `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |


## 설치
### git 사용 (다이렉트 설치)

`pip` 명령어를 사용해 `ember`(필요한 기능) 설치하십시오.

```
pip install git+https://github.com/endgameinc/ember.git
```

EMBER feature 추출 기능을 사용하려면, 저장소 복제 후 train model scripts 사용바랍니다.

### EMBER 저장소 복제 후 설치하기
`pip` 또는 `conda` 명령어를 통해 `ember`(필요한 기능) 설치하십시오.

```
pip install -r requirements.txt
python setup.py install
```

```
conda config --add channels conda-forge
conda install --file requirements_conda.txt
python setup.py install
```

## Scripts

`train_ember.py`는 간단한 모델 트레이닝 스크립트입니다. ember features 벡터화 후 LightGBM 모델 학습 가능합니다.

```
python train_ember.py [/path/to/dataset]
```

`classify_binaries.py` 는 PE파일 정확도를 반환해주는 스크립트입니다.

```
python classify_binaries.py -m [/path/to/model] BINARIES
```

## 내보내기

raw 데이터(feature)는 모델 트레이닝을 위해 벡터화된 내용을 metadata 형식으로 확장(저장)할 수 있습니다. 2가지 함수는 다음처럼 추가 파일을 생성합니다:

```
import ember
ember.create_vectorized_features("/data/ember2018/")
ember.create_metadata("/data/ember2018/")
```

파일 생성되면, 다음과 같은 편리한 함수를 통해 데이터를 읽을 수 있습니다:

```
import ember
X_train, y_train, X_test, y_test = ember.read_vectorized_features("/data/ember2018/")
metadata_dataframe = ember.read_metadata("/data/ember2018/")
```

ember 모듈 설치 및 데이터 다운 후, 다음과 같이 간단한 벤치마크 ember model 재현 가능합니다:

```
import ember
ember.create_vectorized_features("/data/ember2018/")
lgbm_model = ember.train_model("/data/ember2018/")
```

모델 학습이 끝나면, ember 모듈은 PE파일 정확도(악성코드)를 예측(탐지)할 수 있습니다:

```
import ember
import lightgbm as lgb
lgbm_model = lgb.Booster(model_file="/data/ember2018/ember_model_2018.txt")
putty_data = open("~/putty.exe", "rb").read()
print(ember.predict_sample(lgbm_model, putty_data))
```

## 인용

만약 당신이 이 오픈소스(데이터)를 사용하면 이 사이트[(paper)](https://arxiv.org/abs/1804.04637)를 참고바랍니다:
If you use this data in a publication please cite the following :

```
H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

@ARTICLE{2018arXiv180404637A,
  author = {{Anderson}, H.~S. and {Roth}, P.},
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.04637},
  primaryClass = "cs.CR",
  keywords = {Computer Science - Cryptography and Security},
  year = 2018,
  month = apr,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
```
