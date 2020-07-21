<img src="resources/logo.png" align="right" width="250px" height="250px">

# Endgame Malware BEnchmark for Research

EMBER 데이터셋은 PE파일(features) 모음으로 악성코드 연구자를 위해 벤치마크 데이터셋을 제공합니다. EMBER2017 데이터셋은 2017년까지 스캔된 110만개 PE 파일(features)이며, EMBER2018 데이터셋은 2018년까지 1백만개 PE파일(features)입니다. 이 저장소를 통해 벤치마크 모델을 쉽게 재현하고, 제공된 데이터셋을 확장해 벤치마크 모델을 새롭게 PE 파일(hwp, pdf, jpg 등)로 분류할 수 있습니다.

이 링크는 데이터셋 관련 자세한 논문입니다: [https://arxiv.org/abs/1804.04637](https://arxiv.org/abs/1804.04637)

## 기능

[LIEF](https://lief.quarkslab.com/) 프로젝트는 EMBER 데이터셋 PE파일(features)을 추출하는데 사용됩니다. LIEF 프로젝트는 EMBER 데이터셋 PE파일 features 을 추출하는데 사용됩니다. Raw 파일(features)은 JSON 포맷으로 추출된다. 그리고 공개적으로 이용 가능한 데이터셋으로 포함됩니다. Vectorized(features)은 Raw(features)으로부터 생성됩니다. CSV, dataframe 등은 바이너리 포맷으로 저장됩니다. 이 저장소를 사용하면 PE파일을 Raw(features)와 Vectorized(features)로 쉽게 생성할 수 있습니다. 연구자는 자체 기능을 구현하거나 기존 기능과 다르게 벡터화 할 수 있습니다.

feature calculation version 입니다. Feature version 1 은 LIEF library version 0.8.3 으로 계산됩니다. Feature version 2 는 additional data directory feature, updated ordinal import processing 포함되고 LIEF library version 0.9.0 으로 계산됩니다. 윈도우/리눅스에서 LIEF version 0.10.1 으로 두 환경 모두 Feature version 2 작동됨을 확인했습니다.

## 년도별

The first EMBER dataset consisted of version 1 features calculated over samples collected in or before 2017. The second EMBER dataset release consisted of version 2 features calculated over samples collected in or before 2018. In conjunction with the second release, we also included the version 2 features from the samples collected in 2017. Combining the data from 2017 and 2018 will allow longer longitudinal studies of the evolution of features and PE file types. But different selection criteria were applied when choosing samples from 2017 and 2018. Specifically, the samples from 2018 were chosen so that the resultant training and test sets would be harder for machine learning algorithms to classify. Please beware of this inconsistancy while constructing your multi-year studies.

첫 번째, EMBER 데이터셋은 2017년까지 수집된 샘플을 계산해 version 1 features 구성했습니다.  두 번째, EMBER 데이터셋은 2018년까지 수집된 샘플을 계산해 version 1 features 구성했습니다. EMBER2017 version 2 는 EMBER2017 version 1 까지 포함했습니다. 2017년부터 2018년까지 데이터를 조합해 사용하면 PE파일(features)의 장기적인 유형 진화에 대해 연구 가능합니다. 하지만, 2017년과 2018년 샘플을 선택할 때 기준을 잘 확인해야 합니다. 특히, 2018년 샘플을 선택하면 머신러닝 알고리즘으로 test sets, resultant training 하기 어려워집니다. 이 부분은 몇 년간 연구했지만 양해바라며 유의바랍니다.

## 다운로드

여기서 데이터를 다운로드 하세요:

| Year | Feature Version | Filename                     | URL                                                                                                                              | sha256                                                             |
|------|-----------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 2017 | 1               | ember_dataset.tar.bz2        | [https://pubdata.endgame.com/ember/ember_dataset.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset.tar.bz2)               | `a5603de2f34f02ab6e21df7a0f97ec4ac84ddc65caee33fb610093dd6f9e1df9` |
| 2017 | 2               | ember_dataset_2017_2.tar.bz2 | [https://pubdata.endgame.com/ember/ember_dataset_2017_2.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset_2017_2.tar.bz2) | `60142493c44c11bc3fef292b216a293841283d86ff58384b5dc2d88194c87a6d` |
| 2018 | 2               | ember_dataset_2018_2.tar.bz2 | [https://pubdata.endgame.com/ember/ember_dataset_2018_2.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset_2018_2.tar.bz2) | `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |


## 설치
### Instrall directly from git
Use `pip` to install the `ember` and required files

```
pip install git+https://github.com/endgameinc/ember.git
```

This provides access to EMBER feature extaction for example.  However, to use the scripts to train the model, one would instead clone the repository.


### Install after cloning the EMBER repository
Use `pip` or `conda` to install the required packages before installing `ember` itself:

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

The `train_ember.py` script simplifies the model training process. It will vectorize the ember features if necessary and then train the LightGBM model.

```
python train_ember.py [/path/to/dataset]
```

The `classify_binaries.py` script will return model predictions on PE files.

```
python classify_binaries.py -m [/path/to/model] BINARIES
```

## Import Usage

The raw feature data can be expanded into vectorized form on disk for model training and into metadata form. These two functions create those extra files:

```
import ember
ember.create_vectorized_features("/data/ember2018/")
ember.create_metadata("/data/ember2018/")
```

Once created, that data can be read in using convenience functions:

```
import ember
X_train, y_train, X_test, y_test = ember.read_vectorized_features("/data/ember2018/")
metadata_dataframe = ember.read_metadata("/data/ember2018/")
```

Once the data is downloaded and the ember module is installed, this simple code should reproduce the benchmark ember model:

```
import ember
ember.create_vectorized_features("/data/ember2018/")
lgbm_model = ember.train_model("/data/ember2018/")
```

Once the model is trained, the ember module can be used to make a prediction on any input PE file:

```
import ember
import lightgbm as lgb
lgbm_model = lgb.Booster(model_file="/data/ember2018/ember_model_2018.txt")
putty_data = open("~/putty.exe", "rb").read()
print(ember.predict_sample(lgbm_model, putty_data))
```

## Citing

If you use this data in a publication please cite the following [paper](https://arxiv.org/abs/1804.04637):

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
