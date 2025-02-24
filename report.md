# Questions

<br>

## Question 1. Explain in detail what type of fine-tuning method you applied. Explain why the proposed fine-tuning method is suitable for the few-shot train set scenario.

사전학습된 OWL-ViT 모델의 vision, text encoder 부분을 동결(freezing)하고 detection heads의 가중치 파라미터만을 업데이트하는 방식을 적용하였습니다. 

구체적으로는 OWL-ViT 모델의 전체 레이어 중 `vision`, `text` 키워드가 레이어 이름에 포함되는 backbone 부분을 미세조정에서 제외하고, detection heads에 해당하는 레이어들 (`class_head`, `box_head`, `objectness_head`, etc.) 중 일부를 선택하고 학습 및 테스트를 반복하였습니다.

손실함수는 분류, L1, GIoU 손실함수 등을 가중합하여 사용하였습니다. 

단일 클래스 상황이므로 분류에 관한 Score는 잘 작동하는 것으로 보이는 경우가 많았으므로, 분류 손실의 가중치를 매우 낮게 설정하였습니다. (단, 모델의 성능이 매우 저조하므로 정말로 분류가 잘 학습된 것인지는 확신하기 어렵습니다.)

Bounding box localization에 관한 L1, GIoU 손실함수에 더 높은 가중치를 부여하여 여러 번 학습을 시도하였습니다. 경험적으로 보았을 때, L1 손실함수만을 사용하는 경우, 모델이 예측하는 바운딩박스의 크기가 Groun truth와 비슷해지지만, 위치가 잘 일치하지 않으므로 실제로 IoU는 대부분 0인 경우가 많았습니다.

이를 보완하기 위해 L1 손실함수 뿐만 아니라 GIoU 손실함수를 함께 가중합하여 사용해 보았습니다. 이 경우, 모델이 예측하는 바운딩 박스의 크기가 Ground truth에 비해 매우 커지는 경향이 있었습니다. 

추측으로는 GIoU 손실에 의해 모델이 학습하는 과정에서 모델의 예측이 바운딩 박스 좌표에 가까워지는 것보다는, 모델이 예측하는 바운딩박스의 크기를 늘림으로써 GIoU 계산시의 최대 바운딩박스 크기를 늘리고, 이를 통해 정답 바운딩박스를 포함하려는 크게 작용했던 것으로 보입니다. 

이에 대한 방안으로 적절한 가중치 조합을 시도해보거나, 보다 개선된 IoU 관련 손실함수, 예를 들면 Distance-IoU 또는 Complete-IoU 등을 시도해볼 수 있었으나 시간상의 문제로 하지 못했습니다.

<br>

### 시도한 방법이 Few-shot 데이터셋 상황에 적합하다고 생각한 이유

극도로 적은 데이터셋을 갖고 있는 상황이므로, 전체 모델을 미세조정하려는 경우에는 과적합될 가능성이 높을 것입니다. Pretrained OWL-ViT 모델의 Encoder의 일반화된 representation 능력을 그대로 유지하고, 출력 단계만을 조정하는 것이 직관적일 것으로 생각했습니다.

모든 학습 이미지 안에는 하나의 객체 정보만이 있고, 모든 객체의 라벨은 `STABBED`로 동일하므로 분류 관련 파라미터의 학습 비중은 낮다고 판단했습니다. 바운딩박스 Localization 학습이 핵심으로 보이며, 반복적인 실험 결과에도 불구하고 매우 저조한 성능을 보입니다.

<br>

### Localization이 잘 되지 않는 이유에 관한 개인적인 생각

주어진 20장의 학습 이미지와 그 안의 단일 객체 정보만이 있는 경우, 평범하게 미세조정을 하는 방식으로는 모델로 하여금 다양한 바운딩박스의 위치, 크기, 모양 등을 학습시키기 어려울 것으로 보입니다. 

기본적으로 few-shot finetuning 상황이기 때문에 딥러닝 모델의 특성상 fine-tuning 방식을 변경해도 (예를 들면, multi-stage fine-tuning 방식을 적용해도) 실제로 활용할 수 있을 정도의 객체탐지 성능을 얻기 어려울 것으로 보입니다. 이는 OWL-ViT 모델의 한계가 아니라 딥러닝 모델의 한계에 의한 것입니다.

따라서 20장의 이미지를 학습 이미지로 삼아 직접적으로 모델을 미세조정하는 것보다는, 일종의 reference image로 참고하며 모델로 하여금 유사한 패턴을 추론 단계에서 찾도록 하는 방법이 더 적합할 것으로 보입니다. 객체탐지 분야에 대해서는 많은 지식이 없지만, 이와 유사하게 이상탐지 분야에서 CLIP Encoder를 이용하여 Few-shot inference를 수행하는 WinClip 모델의 사례를 참고하여 프로젝트를 진행한다면 흥미로울 것 같습니다.

<br>

## Question 2. Explain carefully the limitation of the proposed method and OWL-ViT architecture for the given task

<br>

### 제시된 방법의 한계

* 미세조정 이후에도 모델이 예측하는 바운딩 박스는 실제 Ground truth를 잘 반영하지 못했습니다. 즉, 바운딩박스가 훨씬 크게 출력되거나, 비슷한 크기의 바운딩박스가 예측되어도 실제 바운딩박스와 잘 일치하는 경우가 없었습니다. 그러므로 IoU가 매우 낮아지고, 최종 평가지표인 Recall, Average Precision을 계산하는 단계에서 수치가 0이 되었습니다.

전반적으로 OWL-ViT 모델과 같은 Zero-shot 객체탐지 분야에 관한 이해, 그리고 Pretrained encoder의 representation을 잘 활용하는 미세조정 방법에 대해 미숙했습니다.

<br>

### OWL-ViT 모델 구조의 한계

반복적인 학습에도 모델의 미세조정 성능이 좋지 않았기 때문에 OWL-ViT 모델의 Encoder가 산업용 이미지와 같은 데이터셋에 대해 적절한 representation을 생성하기 어려운가를 생각해볼 수 있습니다.

다만, 이상탐지 분야에서 `PaDiM`, `PatchCore` 모델의 경우와 같이 Pretrained backbone의 계층별 특성을 잘 활용하여 특정 데이터셋에 대한 역전파 학습 없이 우수한 이상탐지 성능을 보이는 사례가 있었으므로 Encoder의 성능에 대해서는 잘 판단하기 어렵습니다.

모델 자체의 구조적인 한계로 지적할 만한 부분은 거의 없다고 생각되며, Object detection 분야의 few-shot finetuning 관련 지식이 부족했던 점이 더 크게 작용했을 것으로 보입니다.

<br>

### 개인적인 의견

만약 주어진 문제가 metal bearing dataset의 표면 정보로부터 `STABBED`와 같은 단일 패턴을 감지하는 것, 또는 여러 객체(이상)를 탐지하더라도 Anomaly Classification과 같은 이상 패턴의 분류가 덜 중요한 상황이라면 비지도학습 방식으로 이상탐지 모델을 적용해보는 것도 흥미로울 것으로 보입니다. 예를 들면, 이상탐지 분야에서 Zero-shot Inference 성능을 갖고 있는 `WinClip` 모델을 활용해보는 것도 좋아 보입니다. 이 경우, 주어진 20장의 학습용 이미지를 WinClip 모델의 k-reference normal image로 활용해볼 수도 있을 것 같습니다.

---
