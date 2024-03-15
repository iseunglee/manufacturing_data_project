# manufacturing_data_project

# manufacturing_data_project
23년 여름 한국데이터산업진흥원에서 개최한 데이터청년캠퍼스에 참가하여 클린룸 장비를 생산하는 제조기업의 문제를 해결하는 프로젝트를 진행했었습니다. 당시, 저와 팀원 모두 개발과 관련된 역량이 부족하여 웹 상에 배포하지 못한 채로 프로젝트를 마감했었습니다. 그때의 아쉬움을 해소하고자 진행했던 프로젝트를 고도화하는 방향으로 이번 프로젝트의 방향성을 설정했습니다.

---

## 사용된 데이터
사용된 데이터는 FFU 생산 데이터입니다. FFU란 반도체, 디스플레이, 2차 전지 등 고도의 품질을 요구하는 산업분야 인프라의 핵심을 담당하는 클린룸의 구성 장비 중 하나입니다. FFU는 천장에 설치되어 공기를 필터링하고 순환시키는 역할을 합니다.



![image](https://github.com/iseunglee/manufacturing_data_project/blob/main/image.png)



해당 데이터는 제품의 생산이 완료된 후, 13가지 항목을 측정한 결과를 저장한 데이터입니다. 대외비로 인해 당시 사 측에서 값이 수정된 데이터를 제공받았기 때문에 시각화 결과가 비교적 어색할 수 있다는 점을 미리 알립니다.

---

## 기능 구현
구현할 기능은 데이터 시각화 대시보드, OpenAI API를 이용한 정형데이터 챗봇, 제품 등급 예측 모델 크게 3가지입니다.


1. 데이터 시각화 대시보드

    많은 양의 데이터를 그래프를 통해 한눈에 파악할 수 있도록 시각화합니다.
    구성되는 그래프는 총 4가지입니다.
    - 생산된 FFU의 등급별 개수를 볼 수 있도록 파이차트로 구현합니다.
    - 각 등급의 모터타입, 3상/1상, 필터타입 생산 대수를 파악하도록 막대차트로 구현합니다.
    - 불량품을 파악할 수 있도록(이상치를 파악할 수 있도록) 전력소비량, 노이즈, 진동 항목에 대한 박스플롯을 생성합니다.
    - 높은 풍량을 유지하면서 전력 소비를 최소화하는 장비를 식별하고자 산점도로 표현합니다.




2. OpenAI API를 이용한 정형데이터 챗봇
    자연어로 정형데이터를 파악할 수 있는 챗봇입니다.
    복잡한 쿼리나 프로그래밍 기술 없이도 자연어를 통해 직관적인 데이터 접근을 가능하게 하며, 시간절약 및 효율성이 증가할 것으로 기대합니다.




3. 제품 등급 예측 모델
    당시 사내 프로젝트에서 해결해야 했던 문제 중 하나였습니다. 견적서 입력 시 제품의 등급을 입력해야 했고, 사업팀의 신입사원분들이 해당 업무에 어려움을 겪고 있었습니다. 이를 해결하고자 당시에도 등급 예측 모델을 생성했으나, 웹 상에 배포하진 못했습니다. 이번 기회에 streamlit을 통해 배포하고자 합니다.



---



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