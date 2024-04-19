# manufacturing_data_project
23년 여름 한국데이터산업진흥원에서 개최한 데이터청년캠퍼스에 참가하여 클린룸 장비를 생산하는 제조기업의 문제를 해결하는 프로젝트를 진행했었습니다. 당시, 저와 팀원 모두 개발과 관련 역량이 부족하여 웹 상에 배포하지 못한 채로 프로젝트를 마감했었습니다. 그때의 아쉬움을 해소하고자 진행했던 프로젝트를 고도화하는 방향으로 이번 프로젝트의 방향성을 설정했습니다.


---


## 사용된 데이터
사용된 데이터는 FFU 생산 데이터입니다. FFU란 반도체, 디스플레이, 2차 전지 등 고도의 품질을 요구하는 산업분야 인프라의 핵심을 담당하는 클린룸의 구성 장비 중 하나입니다. FFU는 천장에 설치되어 공기를 필터링하고 순환시키는 역할을 합니다.


<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/blob/main/image.png">
</p>


해당 데이터는 제품 생산이 완료된 후, 13가지 항목에 대한 측정 결과를 저장한 데이터입니다. 대외비로 인해 값이 임의로 수정된 생산 데이터를 사용하였음을 미리 알립니다.


---


## 기능 구현
구현할 기능은 데이터 시각화 대시보드, 랭체인을 활용한 정형데이터 챗봇, 제품 등급 예측 모델 크게 3가지입니다.


1. 데이터 시각화 대시보드

    많은 양의 데이터를 그래프를 통해 한눈에 파악할 수 있도록 시각화합니다.
    구성되는 그래프는 총 4가지입니다.
    - 생산된 FFU의 등급별 비율을 볼 수 있도록 파이차트로 구현합니다.
    - 각 등급의 모터타입, 3상/1상, 필터타입 생산 대수를 파악하도록 막대차트로 구현합니다.
    - 불량품을 파악할 수 있도록(이상치를 파악할 수 있도록) 전력소비량, 노이즈, 진동 항목에 대한 박스플롯을 생성합니다.
    - 높은 풍량을 유지하면서 전력 소비를 최소화하는 장비를 식별하고자 산점도로 표현합니다.

2. 랭체인을 활용한 정형데이터 챗봇
   
    자연어로 정형데이터를 파악할 수 있는 챗봇입니다.
    복잡한 쿼리나 프로그래밍 기술 없이도 자연어를 통해 직관적인 데이터 접근을 가능하게 하며, 시간절약 및 효율성이 증가할 것으로 기대합니다.

3. 제품 등급 예측 모델

    당시 사내 프로젝트에서 해결해야 했던 문제 중 하나였습니다. 견적서 입력 시 제품의 등급을 입력해야 했고, 업무에 적응하지 못한 사업팀의 신입사원분들이 해당 업무에 어려움을 겪고 있었습니다. 이를 해결하고자 당시에도 등급 예측 모델을 생성했으나, 웹 상에 배포하진 못했습니다. 이번 기회에 streamlit을 통해 배포하고자 합니다.
    
    
---


## 상세 설명

### streamlit의 multipage apps
streamlit의 multipage apps를 활용하여 웹 페이지를 구조화했습니다. 앱이 커지면 여러 페이지로 구성하는 것이 유용합니다. 이를 통해 개발자는 앱을 더 쉽게 관리하고 사용자는 더 쉽게 탐색할 수 있습니다. 또한 세션 상태를 초기화하는 코드없이도 페이지를 클릭하면 프런트엔드를 다시 로드하지 않고 세션을 저장하기 때문에 페이지간 이동이 더욱 빨라집니다.
```
📦manufacturing_data_project
 ┣ 📜main.py
 ┣ 📂pages
 ┃ ┣ 📜1_📈_FFU_Dashboard.py
 ┃ ┣ 📜2_💬_FFU_Chatbot.py
 ┃ ┗ 📜3_🤖_Prediction_Model.py
```


 ### 데이터 시각화 대시보드
 <p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/884659f5-af64-4824-8df3-f03f811155f1">
</p>

대시보드를 구성하는 차트는 다음과 같습니다.
1. 생산된 장비의 등급을 알 수 있는 파이차트
2. 각 등급별로 어떤 모터타입(BLDC/AC)을 적용했는지, 3상/1상을 적용했는지, 어떤 필터타입(ULPA/HEPA)을 적용했는지 장비 개수를 볼 수 있는 막대차트
3. 불량품을 확인할 수 있는 전력소비량(power_consumption), noise(소음), vibration(진동)에 대한 박스플롯
4. 높은 풍량을 유지하며 낮은 전력소비량을 띄는 장비를 식별하기 위한 산점도

plotly 라이브러리를 이용하여 인터렉티브한 시각화를 구현했습니다. 선택한 변수에 맞게 그래프를 변경할 수 있고 그래프 내에서도 조작이 가능합니다.



### 랭체인을 활용한 정형 데이터 챗봇
 <p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/09493cb4-89fe-4d12-9ac2-17a8c5ba9dd2">
</p>

랭체인을 활용하여 자연어로 데이터에 대한 인사이트를 추출하도록 했습니다. 이 기능의 장점은 프로그래밍 기술없이 사람이 사용하는 언어로 데이터에 대한 정보를 얻을 수 있다는 점입니다. 실제로 프로젝트가 완성된 후 사내 발표를 진행할 때, 파이썬이나 판다스가 익숙하지 않은 분들에 대한 접근성이 떨어진다는 의견이 있었는데, 해당 기능을 통해 프로그래밍 지식없이도 데이터를 다룰 수 있을 것으로 기대합니다.



### 제품 등급 예측 모델
 <p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/7a46829b-a552-4b3c-99f1-134f7b3a6b04">
</p>

생산 데이터를 통해 학습시켜 생산된 제품의 등급을 자동으로 결정하는 분류 모델을 생성하였습니다. 최종적으로 93%의 정확도를 가지는 LightGBM 모델을 사용하였습니다. 자세한 내용은 아래를 참고해 주시길 바랍니다.

<br>

1. 데이터 정의
<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/59ee40e3-f27a-49ca-a69f-6c2c3189ad4f">
</p>

생산이 완료된 후 장비에 대해 테스트를 진행한 데이터입니다. 총 13가지 항목에 대한 값을 측정합니다. 측정된 결과값을 바탕으로 제품에 대한 등급을 결정합니다. a부터 e등급까지 총 5개 등급이 있습니다. a 등급이 가장 우수한 성능의 제품이며, 로우 데이터엔 등급 컬럼은 존재하지 않았지만 견적서를 바탕으로 직접 모델 학습에 사용될 데이터를 생성했습니다.

<br>

2. 데이터 전처리
<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/e968526f-f727-4d3b-a544-ddca4601d9a8">
</p>
<br>
우선 독립변수들 간 상관관계 분석을 실시했습니다. 차원이 커질수록 성능에 악영향을 끼치기 때문에, 해당 분석을 통해 차원을 조금이라도 줄이고자 했습니다. 시각화를 통해 vibration과 current 변수가 비교적 높은 상관관계를 띄고 있어 제외했습니다. 이는 FFU 장비에 대한 도메인 지식으로만 놓고 봐도 납득 가능한 인사이트였습니다. 제품의 등급이 높아질수록 고사양의 부품이 사용될 것이고 이는 더 많은 전류(current)와 진동수(vibration)를 유발할 것이기 때문입니다. 
<br>
<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/7df9729c-9dae-4cc4-9588-eea55766e01f">
</p>
<br>
또한 오버샘플링을 실시했습니다. d, e와 같이 비교적 낮은 등급에 대한 오버샘플링을 실시하기 전 데이터로 학습을 시킨 후 성능을 살펴보기 위해 혼동행렬 시각화를 진행한 결과 d, e 등급을 잘 분류하지 못한 것을 볼 수 있었습니다. 낮은 등급의 제품들은 생산량이 적어 데이터가 적기 때문이라는 판단을 하였고, 해당 클래스를 오버샘플링 즉, 데이터의 개수를 증가시켜 좀 더 균등한 데이터가 될 수 있도록 했습니다.
<br>

3. Pycaret을 통한 실험 모델 선정
<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/eababaf3-cf29-4dc0-bb09-c96d00409294">
</p>
<br>
파이캐럿이라는 AutoML 라이브러리를 사용하여 좋은 성능을 보이는 모델을 찾고자 했습니다. 프로젝트 준비 기간은 한정되어 있고 모든 분류 모델에 대한 실험을 진행할 수 없었기에 해당 라이브러리를 통해 단 몇 줄의 코드만으로 사용한 데이터에 좋은 성능을 보이는 모델을 찾을 수 있었습니다. 정확도 상위 4개의 모델을 선정하였습니다.
<br>
<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/1ad9da24-f0fe-436b-bf44-324f7b172b14">
</p>
<br>
총 4가지 분류모델(Random Forest, Light GBM, Gradient Boosting Classifier, CatBoost)에 대해 네 종류의 실험을 하여 총 16번의 실험을 진행했습니다.
- BaseModel : 하이퍼파라미터 튜닝 X, 오버샘플링 X 데이터 사용
- Optuna : Optuna 하이퍼파라미터 튜닝 O, 오버샘플링 X 데이터 사용
- SMOTE : Optuna 하이퍼파라미터 튜닝 x, 오버샘플링 O 데이터 사용
- SMOTE Optuna : Optuna 하이퍼파라미터 튜닝 O, 오버샘플링 O 데이터 사용
결과적으로 오버 샘플링을 적용하지 않고, 하이퍼파라미터 튜닝을 진행한 Light GBM 모델이 제일 높은 정확도를 보였고, 해당 모델을 선정했습니다.

<br>

4. 모델링 결과
<p align="center">
  <img src="https://github.com/iseunglee/manufacturing_data_project/assets/162934058/752b0b87-112a-46e7-8451-9da8da8ef3d4">
</p>
<br>
최종 모델의 성능을 혼동행렬로 시각화한 결과, 문제가 되었던 d,e 등급의 제품을 오버샘플링하여 오분류 개수를 1개로 줄여 모델의 정확도를 높일 수 있었음을 확인했습니다. 

---


## 진행 일정

### 24-03-14
- box plot 구현
- bar chart 구현

### 24-03-15
- 오전
    - pie chart 구현
    - scatter 구현
    - README 업데이트

- 오후
    - chatbot 구현
    - Light GBM 모델 구현

### 24-03-17
- 멀티페이지 기능 구현
- 대시보드 페이지 UI 개선
- chatbot 기능 업데이트

### 24-03-18
- 예측모델 업데이트

### 24-03-19
- 전체 코드 리펙토링
- UI 개선
- README 최종 업데이트