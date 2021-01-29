# 비슷한 것들은 비슷한 속성을 갖는다
# 머신러닝은 데이터의 분류에 이 원리를 이용하는데, 데이터를 유사하거나 '가장 가까운' 이웃과 동일한 범주에 배치하는 방법이다.


# 최근접 이웃 분류기를 정의하는 주요 개념과 게으른 학습자로 간주되는 이유
# 거리를 이요한 두 예시의 유사도 측정 방법
# k-NN이라 불리는 유명한 최근접 이웃 분류기 적용 방법



### 최근접 이웃 분류의 이해
# 레이블이 없는 예시를 레이블된 유사한 예시의 클래스로 할당해 분류하는 특징으로 정의할 수 있다.
# 사람은 이전에 경험했던 일과의 비교를 통해 새로운 음식을 식별한다.
# 최근접 이웃 분류에서는 컴퓨터가 현재 상황에 대한 결론을 얻고자 과거의 경험을 되살리는 사람과 같은 회상 능력을 적용한다.

# 정지 영상과 동영상에서 광학 글자 인식과 얼굴 인식을 포함하는 컴퓨터 비전 응용
# 어떤 개인이 영화나 음악을 좋아할 것인지 예측하는 추천 시스템
# 특정 단백질과 질병 발견에 사용 가능한 우전자 데이터의 패턴 인식

# 개념을 정의하기는 어렵지만 보면 뭔지 안다면 최근접 이웃이 적합하다고 말할 수 있다.

# 데이터에 노이즈가 많고 그룹 간에 명확한 구분이 없다면 최근접 이웃 알고리즘은 클래스 경계를 식별하고자 어려움을 겪을 것이다.





## k-NN 알고리즘
# k-최근접 이웃(k-NN, k-nearest neighbors) : k-최근접 이웃 정보를 이용해 레이블이 없는 예시를 분류한다.


# 장점
# 단순하고 효율적이다.
# 기저 데이터 분포에 대한 가정을 하지 않는다.
# 훈련 단계가 빠르다.

# 단점
# 모델을 생성하지 않아 특징과 클래스 간의 관계를 이해하는 능력이 제약을 받는다.
# 적잘한 k의 선택이 필요하다.
# 분류 단계가 느리다.
# 명목 특징과 누락 데이터용 추가 처리가 필요하다.




## 거리로 유사도 측정

# 전통적으로 k-NN알고리즘은 유클리드 거리를 사용한다.
# 두 점을 연결하고자 눈금자를 사용할 수 있는 경우에 측정한 거리다.(최단 직선로를 의미하는 '일직선 거리')
# sqrt((p1 - q1)^2 + (p2 - q2)^2 + ... + (pn - qn)^2)
# p와 q는 비교될 예시이고 n개의 특징을 갖는다.
# p1항은 p의 첫 번째 특징 값이고, q1 항은 예시 q의 첫 번째 특징 값이다.



## 적절한 k 선택

# 과적합과 과소적합 사이의 균형 : 편향 분산 트레이드오프

# k를 큰 값으로 선택하면 노이즈가 많은 데이터로 인한 영향이나 분산은 감소하지만
# 작더라도 중요한 패턴을 무시하는 위험을 감수하는 학습자로 편향될 수 있다.

# 훈련 데이터 전체 관측개수만큼 k를 설정하는경우
# 모델은 최근접 이웃과 상관업이 항상 대다수 클래스를 예측

# 한 개의 최근접 이웃을 사용할 경우
# 노이즈가있는 데이터나 이상치가 예시의 분류에 과도한 영향을 미침

# 실제 k의 선택은 학습될 개념의 난이도와 훈련 데이터의 레코드 개수에 의존한다.
# 관례적으로 보통 k를 훈련 예시 개수의 제곱근으로 두고 시작한다.

# k를 좀 더 크게 선택하고, 먼 이웃보다 가까운 이웃의 투표를 좀 더 권위있는 것으로 간주하는 가중 투표 과정을 적용할 수 있다.




## k-NN 사용을 위한 데이터 준비

# 일반적으로 특징은 k-NN 알고리즘에 적용하기 전에 표준 범위로 변환된다.
# 이 과정이 필요한 이유는 거리 공식이 특정한 측정 방법에 매우 의존적이기 때문이다.

# 최소-최대 정규화
# 모든 값이 0에서 1 사이 범위에 있도록 특징을 변환한다.
# X(new) = (X - min(X)) / (max(X) - min(X))

# z-점수 표준화
# X(new) = (X - mean(X)) / sd(X)

# k-NN 훈련 데이터셋에 사용된 재조정 방법이 나중에 알고리즘이 분류할 예시에도 동일하게 적용되야 한다.

# 유클리드 거리 공식은 명목 데이터에는 정의되지 않는다.
# 따라서 명목 특징 간에 거리를 계산하고자 특징을 수치 형식으로 변환할 필요가 있다. : 더미 코딩
# 더미 인코딩은 원핫 인코딩으로도 알려져 있다.

# 명목 특징이 순위인 경우 더미 코딩의 대안으로 범주에 번호를 매기고 정규화를 적용한다. : *범주 간격이 동일한 경우에만 사용해야 한다.




## k-NN 알고리즘이 게으른 이유
# 기술적으로 말해 추상화가 일어나지 않기 때문이다.
# 학습의 정의를 엄격하게 적용하면 게으른 학습자는 실제 어떤 것도 학습하지 않는다.
# 대신 훈련 데이터 글자를 그대로 저장하기만 한다. 따라서 훈련 단계가 매우 빠르게 일어나는데, 실제 훈련 단계에서는 아무것도 훈련하지 않는다.

# 인스턴스기반 학습 또는 암기 학습 이라고도 한다.
# 인스턴스 기반 학습자는 모델을 만들지 않기 때문에 비모수 학습 방법의 부류라고 말할 수 있다.








##### 예제 : k-NN 알고리즘으로 유방암 진단

### 1단계 : 데이터 수집

# 유방암 데이터에는 569개의 암 조직 검사 예시가 들어있으며, 각 예시는 32개의 특징을 갖는다.
# 32개의 특징은 식별 번호와 암 진단, 30개의 수치로 평가된 실험실 측정치로 돼 있다.
# 진단은 악성을 나타내는 M 이나 양성을 나타내는 B로 코드화돼 있다.
# 30개의 수치 측정치는 디지털화된 세포핵의 10개 특성에 대한 평균, 표준 오차, 최악의 값(최댓값)으로 구성된다.

# 반지름(Radius) 
# 질감(Texture) 
# 둘레(Perimeter) 
# 넓이(Area) 
# 매끄러움(Smoothness) 
# 조밀성(Compactness) 
# 오목함(Concavity)
# 오목점(Concave points)
# 대칭성(Symmetry)
# 프랙탈 차원(Fractal dimension)







### 2단계 : 데이터 탐색과 준비

wbcd <- read.csv("data/wisc_bc_data.csv")
str(wbcd)

# id 변수는 단순히 데이터 내에서의 환자 아이디이기 때문에 유용한 정보를 제공하지 않으며 모델에서 제외할 필요가 있다.
# 머신러닝 방법에 상관없이 id변수는 항상 제외시켜야 한다.
wbcd <- wbcd[-1]

# diagnosis는 예측하려는 결과이므로 특별히 관심이 있다.
# 이 특징은 예시가 양성 종양인지 음성 종양인지 여부를 나타낸다
table(wbcd$diagnosis)

# 많은 R 머신러닝 분류기는 목표 특징이 팩터로 코딩돼야만 한다.
# 따라서 diagnosis변수를 다시 코드화 해야한다.
# 또한 labels파라미터를 이용해 "B"와 "M" 값에 유용한 정보를 주는 레이블을 제공할 것이다.
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))

round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

# 설명을 위해 남은 특징 중 세 개만 자세히 살펴보자.
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# 이 값을 살펴보면 값에 문제가 있다는 것을 알 수 있다.
# k-NN의 거리 계산 입력 특징은 측정하는 척도에 상당히 종속된다는 것을 기억하자.


## 변환 : 수치 데이터 정규화
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

normalize(c(1,2,3,4,5))
normalize(c(10, 20, 30, 40, 50))


# lapply(list, function) : 리스트를 취해 각 리스트 항목에 지정된 함수를 적용한다.
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize)) # 2열부터 31열까지 normalize()함수를 적용하고, 결과 리스트를 데이터 프레임으로 변환해 wbcd_n이라는 이름에 할당한다.
summary(wbcd_n)


## 데이터 준비 : 훈련 및 테스트 데이터셋 생성
# 학습자가 레이블이 없는 데이터의 잡합에 대해 얼마나 잘 수행되는가?
# k-NN 모델을 구축하고자 사용되는 훈련 데이터셋과
# 모델의 예측 정확도를 평가하고자 사용되는 테스트 데이터셋
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

# 데이터셋을 만들때는 랜덤 샘플링 방법이 필요하다. 위의 데이터는 이미 무작위로 정렬시켜둔 데이터라서 생략한다.


# k-NN모델을 훈련하고자 diagnosis를 팩터 벡터에 저장하고 훈려 데이터셋과 테스트 데이터셋으로 분리한다.
# 분류기를 훈련하고 평가하는 다음 단계에서 다음 레이블 벡터를 사용할 것이다.
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]








### 3단계 : 데이터로 모델 훈련

# k-NN 알고리즘은 훈련 단계에서 모델을 실제 구축하지는 않는다.
# k-NN과 같은 게으른 학습자를 훈련하는 과정은 단순히 입력 데이터를 구조화된 형식으로 저장하는 것이다.

# 테스트 인스턴스를 분류하고자 분류를 위한 기본 R 함수를제공하는 class 패키지의 k-NN 구현을 사용할 것이다.
# install.packages("class")
require(class)

# class 패키지의 knn() 함수는 대표적인 k-NN 표준 알고리즘을 구현한다.
# 이 함수는 테스트 데이터의 각 인스턴스별로 유클리드 거리를 이용해 k-최근접 이웃을 식별한다.(k는 사용자 지정 숫자)

# p <- knn(train, test, class, k)
# train : 수치 훈련 데이터를 포함하는 데이터 프레임
# test : 수치 테스트 데이터를 포함하는 데이터 프레임
# class : 훈련 데이터의 각 행에 대한 클래스를 갖는 팩터 벡터
# k : 최근접 이웃의 개수를 가리키는 정수

# 훈련 데이터에 469개의 인스턴스를 포함하므로 대략 469개의 제곱근과 동일한 홀수인 k = 21로 시도(2-범주 결과이므로 홀수를 사용해 동점표로 끝날 가능성을 제거)

wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, 
                      cl = wbcd_train_labels, k = 21)







### 4단계 : 모델 성능 평가
# 이 과정의 다음 단계는 wbcd_test_pred 벡터에 있는 예측된 클래스가 wbcd_test_labels 벡터에 있는 실제 값과 얼마나 잘 일치하는가를 평가하는 것이다.

# gmodels 패키지의 CrossTable() 함수 이용
require(gmodels)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = F)

# 이 표의 셀 백분율은 네 범주에 속하는 값의 비율은 나타낸다.

# 좌측 상단 셀은 TN(참 부정)결과를 나타낸다.
# 100개의 값 중 61개는 종양이 양성이고, k-NN 알고리즘이 정확히 양성으로 식별한 경우다.

# 우측 하단 셀은 TP(참 긍정)결과를 나타내며, 분류기와 임상적으로 판단된 레이블이 종양이 악성이라는 것에 동의한다.
# 100개의 예측에서 37개는 모두 TP다.

# 다른 대각에 있는 셀들은 k-NN 예측이 실제 레이블과 일치하지 않는 예시의 개수를 지닌다.

# 좌측 하단의 2개의 예시는 FN(거짓 부정)결과다.
# 이 경우 예측된 값이 양성이지만, 실제로는 종양이 음성이다.
# 이 방향의 오류는 환자가 암이 없다고 믿게되지만, 실제로 병이 확산될 수 있기 떄문에 대가가 엄청나게 클 수 있다.

# 우측 상단의 셀은 값이 존재한다면 FP(거짓 긍정)결과를 포함한다.
# 이 값은 모델이 종양을 악성으로 분류하지만, 실제 양성이 경우에 발생한다.
# FN의 결과보다 덜 위함하지만, 추가 검사나 치료를 받아야 하기 때문에 환자에게 의료비 부담이나 스트레스를 더해줄 수 있으므로 피해야만 한다.








### 5단계 : 모델 성능 개선

# 이전 분류기에 두 가지 간단한 변형을 시도할 것이다.
# 첫째, 수치 특징을 재조정하고자 다른 방법을 사용한다.
# 둘째, k에 몇 가지 다른 값을 시도해 본다.



## 변환 : z-점수 표준화

# 벡터를 표준화 하고자 R 내장함수 scale()을 이용해 기본 방식인 z-점수 표준화 방식으로 값을 재조정한다.
# scale() 함수는 데이터 프렝미에 직접 적용할 수 있으므로 lapply()함수를 사용할 필요가 없다.
wbcd_z <- as.data.frame(scale(wbcd[-1]))
summary(wbcd_z)

wbcd_train_z <- wbcd_z[1:469, ]
wbcd_test_z <- wbcd_z[470:569, ]
wbcd_train_labels_z <- wbcd[1:469, 1]
wbcd_test_labels_z <- wbcd[470:569, 1]
wbcd_test_pred_z <- knn(train = wbcd_train_z, test = wbcd_test_z,
                        cl = wbcd_train_labels_z, k = 21)
CrossTable(x = wbcd_test_labels_z, y = wbcd_test_pred_z,
           prop.chisq = F)

# 모델의 성능이 더 악화되었다.


## k의 대체 값 테스트
# k = 21
wbcd_test_pred_21 <- knn(train = wbcd_train, test = wbcd_test, 
                      cl = wbcd_train_labels, k = 21)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_21, prop.chisq = F)

# k = 1
wbcd_test_pred_1 <- knn(train = wbcd_train, test = wbcd_test, 
                      cl = wbcd_train_labels, k = 1)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_1, prop.chisq = F)

# k = 5
wbcd_test_pred_5 <- knn(train = wbcd_train, test = wbcd_test, 
                        cl = wbcd_train_labels, k = 5)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_5, prop.chisq = F)

# k = 11
wbcd_test_pred_11 <- knn(train = wbcd_train, test = wbcd_test, 
                        cl = wbcd_train_labels, k = 11)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_11, prop.chisq = F)

# k = 15
wbcd_test_pred_15 <- knn(train = wbcd_train, test = wbcd_test, 
                        cl = wbcd_train_labels, k = 15)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_15, prop.chisq = F)

# k = 27
wbcd_test_pred_27 <- knn(train = wbcd_train, test = wbcd_test, 
                        cl = wbcd_train_labels, k = 27)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_27, prop.chisq = F)


# 학습자가 미래 데이터에 대해 일반화됐다는 확신이 필요하다면 100명의 환자 집합을 임의로 몇 개 생성해서 그 결과를 반복적으로 다시 테스트할 수 있다.

