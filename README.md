# ProblemSolving

* 이 저장소는 과제전형을 위해 임시로 생성되었습니다. 2025년 2월 28일 금요일 이후에는 삭제될 예정입니다.

<br>

## How to set conda environment

``` Shell
conda create --name problem_solving python==3.10 --yes
conda activate problem_solving

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install matplotlib==3.9.0
pip install numpy==1.24.4
pip install scipy
```

Or

``` Shell
conda env create -file environment.yaml
```

---
