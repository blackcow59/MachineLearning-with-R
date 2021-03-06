# 4장에서는 일기예보와 거의 동일한 방식으로 확률을 사용하는 나이브 베이즈 알고리즘을 다룬다.

# 확률의 기본 원리
# R 로 텍스트 데이터를 분석하는 데 필요한 특화된 방법과 데이터 구조
# 나이브 베이즈를 이용한 SMS 스팸 메시지 필터의 구축 방법




##### 나이브 베이즈의 이해

# 베이지안 기법 기반의 분류기는 훈련 데이터를 활용해 특징 값이 제공하는 증거를 기반으로 결과가 관측될 확률을 계산한다.
# 나중에 분류기가 레이블이 없는 데이터에 적용될 때 결과가 관측될 확률을 이용해서 새로운 특징에 가장 유력한 클래스를 예측한다.

# 스팸 이메일 필터링과 같은 텍스트 분류
# 컴퓨터 네트워크에서 침입이나 비정상 행위 탐지
# 일련의 관찰된 증상에 대한 의학적 질병 진단.

# 대표적으로 베이지안 분류기는 결과에 대한 전체 확률을 추정하고자 동시에 여러 속성 정보를 고려해야만 하는 문제에 가장 적합하다.
# 많은 머신러닝 알고리즘이 영향력이 약한 특징은 무시하지만
# 베이지안 기법은 가용한 모든 증거를 활용해 예측을 절묘하게 바꾼다.
# 즉, 상당수의 특징이 개별적으로는 모두 상대적으로 미미한 영향만 미치더라도 그 영향을 베이즈 모델에서 모두 결합하면 꽤 큰 영향을 끼칠 수 있다는 의미가 된다.




### 베이지안 기법의 기본 개념

# 베이지안 확률의 이론은 사건에 대한 우도는 복수 시행에서 즉시 이용할 수 있는 증거를 기반으로 해서 추정해야만 한다는 아이디어에 뿌리를 두고 있다.
# 베이지안 기법은 고나측 데이터에서 사건의 확률을 추정하는 방법에 통찰력을 제공한다.



### 확률의 이해

# 사건의 확률은 관측 데이터에서 사건이 발생한 시행 횟수를 전체 시행 횟수로 나눠서 추정한다.
# 시행을 하면 항상 어떤 결과가 발생하기 때문에 가능한 모든 시행 결과의 확률은 항상 1로 합산돼야 한다.

# 두 사건이 동시에 발생할 수 없고 유일하게 가능한 결과임을 의미하는 상호 배타적이고 포괄적인 사건
# 사건은 자신의 여집합과 항상 상호 배타적이고 완전하다.




### 결합 확률의 이해

# 가끔 같은 시행에서 상호 배타적이지 않은 여러 사건을 관찰하는 데 관심이 있다.
# 어떤 사건이 관심있는 사건과 함께 발생한다면 예측을 하고자 해당 사건을 이용할 수 있다.
# 교집합의 확률은 두 사건의 결합 확률에 따라 달라진다.

# 두 사건이 완전히 관련이 없다면 독립 사건 이라고 한다.
# 독립 사건이 동시에 발생할 수 없는 것은 아니다.
# 사건 독립성은 단순히 한 사건의 결과를 아는 것으로는 다른 사건의 결과에 대한 어떤 정보도 제공하지 못한다는 것을 의미한다.

# 종속 사건이 예측모델링의 기반이 된다.
# 종속 사건의 확률을 계산하는것은 독립 사건의 확률을 계산하는 것보다 복잡하다.




### 베이즈 정리를 이용한 조건부 확률 계산

# 종속 사건 간의 관계는 베이즈 정리를 이용해 설명할 수 있으며, 이는 다른 사건이 제공하는 증거를 고려해 한 사건에 대한 확률 추정을 어떻게 바꿀지에 대해 사고하는 방식을 알려준다.

# 한 가지 공식은 다음과 같다.
# P(A|B) = P(A and B) / P(B) : 사건 B가 발생한 경우 사건 A의 확률 : 조건부 확률
# 정의에 따라 P(A and B) = P(A|B) * P(B)
# P(A and B) = P(B and A) 이므로
# P(A and B) = P(B|A) * P(A)
# P(A|B) = P(A and B) / P(B) = P(B|A) * P(A) / P(B)
 
# 베이즈 이론의 실제 작동 방법을 잘 이해하고자 가상의 스팸 필터를 살펴보자
# 받은 메시지의 내용을 모르른 상태에서 메시지의 스팸 상태를 가장 잘 예측하는 것은 이전 메시지가 스팸일 확률은 P(스팸)이다.
# 이 추정을 사전 확률 이라고 한다.

# 단어 비아그라가 나타났던 빈도를 관측하고자 이전에 받았던 메시지를 좀 더 주의깊게 살펴보자
# 이전 스팸 메시지에서 단어 비아그라가 사용된 확률 P(비아그라|스팸)을 우도라고 부른다.
# 비아그라가 어떤 메시지라도 나타날 확률P(비아그라)를 주변 우도 라고 한다.

# 이 증거에 베이즈 이론을 적용해 메시지가 스팸이 될 확률을 측정한 사후 확률을 계산할 수 있다.
# 사후 확률이 50%보다 크다면 메시지는 햄보다 스팸이 될 가능성이 좀 더 크며, 이런 메시지는 걸러져야 한다.




### 나이브 베이즈 알고리즘

# 나이브 베이즈 알고리즘은 분류 문제에 베이즈 이론을 적용할 수 있는 단순한 방법을 정의해준다.
# 나이브 베이즈는 베이지안 기법을 활용하는 유일한 머신러닝 방법은 아니지만 가장 일반적인 방법이다.

# 장점
# 간단하고 빠르고 매우 효율적이다.
# 노이즈와 누락 데이터를 잘 처리한다.
# 훈련에는 상대적으로 적은 예시가 필요하지만, 대용량의 예시에도 매우 잘 작동된다.
# 예측용 추정 확룰을 쉽게 얻을 수 있다.

# 단점
# 모든 특징이 동등하게 중요하고 독립이라는 가정이 잘못된 경우가 자주 있다.
# 수치 특징이 많은 데이터셋에는 이상적이지 않다.
# 추정된 확률이 예측된 클래스보다 덜 신뢰할만한다.

# 나이브 베이즈는 데이터셋의 모든 특징이 동등하게 중요하고 독립적이라고 가정한다.
# 이런 가정은 대부분의 실제 응용에는 거의 맞지 않다.




### 나이브 베이즈를 이용한 분류

# 나이브 베이즈 분류 알고리즘 공식은 다음 공식으로 요약될 수 있다.
# F1에서 Fn까지 특징이 제공하는 증거가 있을 때 클래스 C에서 레벨 L의 확률은 클래스 레벨을 조건으로 하는 각 증거에 대한 확률과 클래스 레벨의 사전 확률, 우도 값을 확률로 변환하는 배율 요소 1/Z의 곱과 동일하다.

# 빈도표를 구축하는 것에서 시작해서 우도표를 구축하는 데 사용하고, "나이브" 독립 가정에 따라 조건부 확률을 곱한다.
# 최종적으로 각 클래스 우도를 확률로 변환하고자 전체 우도로 나눈다.




### 라플라스 추정량

# 라플라스 추정량은 기본적으로 빈도표의 각 합계에 작은 숫자를 더하는데, 특징이 각 클래스에 대해 발생할 확률이 0이 되지 않도록 보장한다.
# 보통 라플라스 추정량은 1로 설정해서 데이터에 각 클래스 특징 조합이 최소 한 번은 나타나도록 보장한다.






### 나이브 베이즈에서 수치 특성 이용

# 나이브 베이즈는 데이터를 학습하고자 빈도표를 사용하기 때문에 행렬을 구성하는 각 클래스와 특징 값의 조합을 생성하려면 각 특징이 범주형이어야 한다.
# 수치 특성은 값의 범주가 없으므로 앞의 알고리즘은 수치 데이터에 직접 작동되지는 않는다.

# 쉽고 효율적인 해결책 중 하나는 수치 특징을 이산화 하는 것인데
# 간단히 빈 이라고 하는 범주에 숫자를 넣는 것을 의미한다.
# 이런 이유로 이산화는 가끔씩 비닝이라고 부르기도 한다.
# 이 방법은 나이브 베이즈로 작업할 때 일반적인 조건인 훈련 데이터가 대용량인 경우에 이상적이다.


# 수치를 이산화하는 다른 몇 가지 방법이 있다.
# 가장 일반적인 것은 데이터 분포에서 자연스러운 범주나 절단점을 찾고자 데이터를 탐색하는 것이다.









##### 예제 : 나이브 베이즈 알고리즘을 이용한 휴대폰 스팸 필터링

# 나이브 베이즈는 이메일 스팸 필터링에 성공적으로 사용돼 왔기 때문에 SMS 스팸에도 적용될 수 있을 것이다.
# 하지만 이메일 스팸과 비교해서 SMS 스팸은 자동화 필터에 대한 다른 어려움이 있다.
# SMS 메시지는 종종 160개 문자로 제한돼 있어서 메시지가 정크인지 구분할 때 사용되는 텍스트 양이 줄어든다.
# 문자 길이의 제한이 작은 휴대폰 키보드와 합쳐지면 많은 사람들이 SMS 속기 형태의 용어를 쓰게 되고, 이로 인해 합법적인 메시지와 스팸 간의 구분이 더욱 모호해졌다.





### 1단계 : 데이터 수집

# 이 데이터셋에는 SMS 메시지의 텍스트와 원치 않는 메시지인지를 나타내는 레이블이 함께 들어있다.
# 나이브 베이즈 분류기는 SMS 메시지가 스팸이나 햄의 프로파일에 잘 맞는지 판단하고자 단어 빈도 패턴을 이용한다.
# 나이브 베이즈 분류기는 메시지의 모든 단어가 제공하는 증거를 감안해 스팸과 햄의 확률을 계산한다.





### 2단계 : 데이터 탐색과 준비

# 분류기를 구축하기 위한 첫 번째 단계는 분석을 하고자 원시 데이터를 처리하는 것이다.
# 텍스트 데이터는 준비하기가 까다로운데, 단어와 문장을 컴퓨터가 이해할 수 있는 형식으로 변환해야 하기 때문이다.
# 데이터를 변환하기 위한 표현으로 단어의 순서는 무시하거 단순히 단어의 출현 여부를 표시하는 변수를 제공하는 단어 주머니를 사용할 것이다.

sms_raw <- read.csv("data/sms_spam.csv")
str(sms_raw)

sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)






## 데이터 준비 : 텍스트 데이터 정리 및 표준화

# SMS 메시지는 단어, 공백, 숫자, 구두점으로 구성된 텍스트 문자열이다. 이런 종류의 복잡한 데이터를 다루려면 많은 생각과 노력이 필요하다.
# 숫자와 구두점 제거 방법, and, but, or 같은 관심 없는 단어의 처리 방법과 문장을 개별 단어로 나누는 방법을 고려할 필요가 있다.
# 다행이도 이 기능은 R의 tm이라는 텍스트 마이닝 패키지로 제공하고 있다.
# install.packages("tm")

require(tm)

# 텍스트 데이터 처리의 첫 단계는 텍스트 문서의 모음인 코퍼스를 생성하는 것이다.
# 문서는 개별 뉴스 기사에서 책이나 웹 페이지 또는 전체 책까지 짧을 수도 있고 길 수도 있다.
# 이 경우 코퍼스는 SMS 메시지 모음이 될 것이다.

# 코퍼스를 생성하려면 휘발성 코퍼스를 참조하는 tm 패키지의 VCorpus()함수를 사용한다.
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# readerControl 파라미터를 지정하면 PDF 파일이나 MS 워드 파일과 같은 출처에서 텍스트 가져오기를 할 수 있다.

print(sms_corpus)

# tm 코퍼스는 근본적으로 복합 리스트이기 때문에 코퍼스에서 리스트 연산을 사용해 문서를 선택할 수 있다.

# inspect() 함수는 결과의 요약을 보여준다.
inspect(sms_corpus[1:2])

# as.character() 함수를 이용해 실제 메시지 텍스트를 볼 수 있다.
as.character(sms_corpus[[1]])

# 여러 문서를 보려면 sms_corpus 객체의 여러 항목에 as.character()를 적용해야 한다.
lapply(sms_corpus[1:2], as.character)

# 먼저 구두점과 결과를 혼란스럽게 하는 글자들을 제거해 텍스트를 정리하고 표준화할 필요가 있다.
# tm_map()함수는 tm 코퍼스에 변환을 적용할 수 있는 방법을 제공한다.

# 첫 번째 변환은 소문자만 사용하도록 메시지를 표준화하는 것이다.
# 이를 위해 tolower() 함수를 이용한다.
# 이 함수를 코퍼스에 적용하려면 tm 래퍼 함수 content_transformer()를 사용해서 tolower()가 코퍼스에 접근하는 데 사용되는 변환 함수로 취급되게 해야한다.
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

# SMS 메시지에서 숫자를 제거해 정리를 계속해보자.
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# 다음 작업은 to, and, but, or와 같이 채우기 위한 단어를 SMS 메시지에서 제거하는 것이다.
# 이런 용어를 불용어라고 하며, 일반적으로 텍스트 마이닝을 하기 전에 제거한다.
# 불용어 목록을 자체적으로 정의하지 않고 tm 패키지에서 제공하는 stopwords() 함수를 사용할 것이다.(디폴트로 일반 영어의 불용어가 사용된다.)
# 불용어 목록에 나타나는 단어를 제거하기 위해 removeWords() 함수를 사용한다.
sms_corpus_clean <- tm_map(sms_corpus_clean,
                           removeWords, stopwords())

# 정리 과정을 계속 진행하고자 removePunctuation() 변환으로 문자 메시지에서 구두점을 제거할 수 있다.
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

# removePunctuation() 변환은 텍스트에서 구두점 문자를 완전히 제거하는데, 이로 인해 의도치 않은 결과가 야기될 수 있다.
# removePunctuation("hello...world")
# "helloworld"

# removePunctuation()의 디폴트 동작을 피하려면 간단히 구두점 문자를 제거하는 대신 대체하는 사용자 정의 함수를 생성한다.
replacePunctuation <- function(x) {
  gsub("[[:punct:]]+", " ", x)
}
# 기본적으로 이는 x에 있는 구두점 문자를 빈칸으로 대체하기 위해 R의 gsub()함수를 사용한다.
# 이 replacePunctuation() 함수는 다른 변환처럼 tm_map()과 함께 사용된다.


# 다른 텍스트 데이터에 대한 일반적인 표준화인 형태소 분석 과정에서는 단어를 어근 형태로 줄인다.
# 형태소 분석 과정은 learned, learning, learns 과 같은 단어를 받아 접미사를 제거해서 기본적인 형태인 learn으로 변환한다.
# 형태소 분석은 머신러닝 알고리즘이 각 변형에 대한 패턴을 학습하는 대신관련 용어를 동일한 개념으로 취급하게 해준다.

# tm 패키지는 SnowballC 패키지와 통합돼 형태소 분석 기능을 제공한다.
install.packages("SnowballC")
require(SnowballC)

# SnowballC 패키지는 wordStem() 함수를 제공해 문자 벡터에 대해 어근 형태의 동일한 벡터를 반환한다.
wordStem(c("learn", "learned", "learning", "learns"))

# wordStem() 함수를 텍스트 문서의 전체 코퍼스에 적용하고자 tm 패키지는 stemDocument() 변환을 제공한다.
# 이전처럼 tm_map() 함수를 이용해 stemDocument()변환을 코퍼스에 정확히 적용한다.
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)


# 숫자, 불용어, 구두점을 제거하고 형태소 분석을 실행하고 나면 문자 메시지는 지금은 제거된 조각을 분리했던 빈칸과 함께 남는다.
# 따라서 텍스트 정리 과정의 최종단계는 stripWhitespace() 변환을 이용해서 추가 여백을 제거하는 것이다.
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)






## 데이터 준비 : 텍스트 문서를 단어로 나누기

# 원하는 대로 데이터가 처리됐기 때문에 마지막 단계는 메시지를 토큰화 과정을 통해 개별 옹어로 나누는 것이다.
# 토큰은 텍스트 문자열의 한 요소로, 이 경우 토큰은 단어다.

# tm 패키지의 DocumentTermMatrix() 함수는 코퍼스를 가져와 문서 용어 행렬이라고 하는 데이터 구조를 만든다.
# 이때 행렬의 행은 문서(SMS 메시지)를 나타내고, 열은 용어(단어)를 나타낸다.
# 행렬에서 각 셀은 행이 표현하고 있는 문서에서 열이 표현하고 있는 단어가 출현하는 횟수를 저장한다.
# 이 데이터 구조는 희소행렬(대다수의 셀이 0으로 채워져 있다.)이라고 불린다.
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# 사전처리를 수행하지 않았다면 기본 설정을 재지정하는 control 파라미터 옵션 목록을 제공해 사전 처리를 여기서 할 수 있다.
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = T,
  removeNumbers = T,
  stopwords = T,
  removePunctuation = T,
  stemming = T
))

# 이 명령은 앞에서 했던 것과 같은 순서로 SMS 코퍼스에 동일한 사전 처리 단계를 적용한다. 
# 하지만 sms_dtm을 sms_dtm2와 비교하면 행렬의 용어 개수에 약간의 차이를 확인할 수 있다.
sms_dtm
sms_dtm2

# 두 개의 DTM을 동일하게 만들기 위해 기본 불용어 함수를 원래의 변환 함수를 사용하는 자체 함수로 재지정한다. 
# 단순히 stopwords = T를 다음 코드로 대체하라
# stopwods = function(x) {removeWords(x, stopwords())}

# 두 경우의 차이는 텍스트 데이터의 중요한 정리 원칙을 보여준다.
# 즉, 작업의 순서가 중요하다.
# 이를 염두해 두고 과정의 초기 단계가 나중에 어떻게 영향을 줄지를 충분히 생각하는 것이 매우 중요하다.






## 데이터 준비 : 훈련 및 테스트 데이터셋 생성

# 분류기가 테스트 데이터셋의 내용을 보지 못하게 할 필요가 있지만, 데이터가 정리되고 처리된 이후에 분할이 일어나는 것이 중요하다.
# 즉, 훈련 데이터셋과 테스트 데이터셋 모두에 대해 정확히 동일한 준비 단계가 필요하다.

# 데이터를 훈련용 75%와 테스트용25% 두 부분으로 분리하려고 한다.
# SMS 메시지는 임의의 순서로 정렬돼 있으므로, 단순히 처음 4,169개를 훈련용으로 가져오고 나머지 1,390개를 테스트용으로 남겨둔다.
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559, ]

# 편의상 훈련과 테스트 데이터 프레임의 행별 레이블을 갖는 벡터를 각각 저장해두는 것이 좋다.
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

# 이 부분집합이 SMS 데이터 전체 집합을 대표하는지를 확인하고자 훈련 데이터 프레임과 테스트 데이터 프레임의 스팸 비율을 비교해보자.
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))





## 텍스트 데이터 시각과 : 단어 구름(World cloud)

# 단어 구름은 텍스트 데이터에서 단어가 나타나는 빈도를 시각적으로 보여주는 방법이다.
# 텍스트에서 더 자주 나타나는 단어는 더 큰 폰트로 보여주고, 덜 일반적인 용어는 더 작은 폰트로 보여준다.
# 스팸과 햄의 구름을 비교하는 것은 나이브 베이즈 스팸 필터의 성공 여부를 판단하는데 도움이 되므로, SMS 메시지의 단어 유형을 시각화하는데 이 함수를 사용할 것이다.
# install.packages("wordcloud")
require(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, random.order = F)
# ramdom.order = F 이므로 구름은 빈도가 높은 단어가 중심에 더 가깝게 위치하도록 랜덤하지 않은 순서로 배열될 것 이다.
# min.freq 파라미터는 단어가 구름에 보이기 전에 코퍼스에서 나타나는 횟수를 지정한다.

# 좀 더 흥미로운 시각화는 SMS 스팸과 햄 구름을 비교하는 것이다.
# 스팸과 햄을 별도의 코퍼스로 구성하지 않았기 때문에 wordcloud() 함수의 매우 유익한 특징을 사용한다.
# 이 함수는 원시 텍스트 문자열 벡터가 주어지면 구름을 보여주기 전에 공통적인 텍스트 준비 과정을 자동으로 적용한다.

spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

par(mfrow = c(1,2))
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
# max.words 파라미터를 사용해 가장 일반적인 단어 40개를 관찰하기로 설정한다.
# scale 파라미터는 구름에서 단어의 최대, 최소 폰트 크기를 조정하기 위한 것이다.

# 스팸 메시지는 urgent, free, mobile, claim, stop 과 같은 단어를 포함한다.
# 이런 단어는 햄에는 전혀 나타나지 않는다.
# 햄 메시지는 can, sorry, need, time 과 같은 단어를 사용한다.







## 데이터 준비 : 자주 사용하는 단어의 지시자 특성 생성

# 데이터 준비 과정에서 최종 단계는 희소 행렬을 나이브 베이즈 분류기를 훈련시키고자 사용하는 데이터 구조로 변환하는 것이다.
# 현재 희소 행렬은 6,500개 이상의 특징을 포함한다.
# 이 특징은 적어도 하나의 SMS 메시지에 나타나는 모든 단어에 대한 특징이다.
# 특징의 개수를 줄이고자 다섯 개 이하의 메시지에 나타나거나 훈련 데이터에서 약 0.1% 레코드보다 작게 나타나는 단어를 제거할 것이다.

# 자주 나타나는 단어를 찾으려면 tm 패키지에서 findFreqTerms() 함수를 사용하면 된다.
# 이 함수는 DTM을 받아 최소 횟수만큼 나타나는 단어를 포함하는 문자 벡터를 반환한다.
findFreqTerms(sms_dtm_train, 5)

sms_freq_word <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_word)

# 이제 DTM을 필터링해 벡터에 자주 나타나는 용어만 포함하게 할 필요가 있다.
# 앞서 했듯이 DTM의 특정 부분을 요청하려면 데이터 프레임 스타일 연산을 사용하되 열 이름이 DTM에 포함된 단어를 따른다는 점을 주목해야 한다.
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_word]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_word]


# 나이브 베이즈 분류기는 대개 범주형 특정으로 된 데이터에 대해 훈련된다.
# 이 점이 문제가 되는데, 희소행렬의 셀은 숫자이며 메시지에 나타나는 단어의 횟수를 측정하기 때문이다.
# 셀의 값을 단어의 출현 여부에 따라 단순히 예(yes) 또는 아니오(no)를 나타내는 범주형 변수로 바꿀 필요가 있다.

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() 함수는 행렬의 각 행이나 열에 대해 임의의 함수가 호출되게 한다.
# 그리고 행이나 열을 명시하고자 MARGIN 파라미터를 사용한다.(1 = 행, 2 = 열)
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

sms_train
sms_test











### 3단계 : 데이터로 모델 훈련

# 원시 SMS 메시지가 통계 모델에 의해 표현되는 형식으로 변환 됐으므로 나이브 베이즈 알고리즘을 적용할 시점이 됐다.
# 이 알고리즘은 해당 SMS 메시지가 스팸일 확률을 추정하는 데 단어의 존재나 부재를 이용한다.

# 사용할 나이브 베이즈 구현은 e1071 패키지에 있다.
# install.packages("e1071")
require(e1071)

# 3장에서 분류를 위해 사용했던 k-NN 알고리즘과 달리 나이브 베이즈 학습자는 분류를 위해 별도의 단계에서 훈련되고 사용된다.


## 나이브 베이즈 분류 구문(e1071 패키지의 naiveBayes()함수 사용)

# 분류기 구축 :
# m <- naiveBayes(train, class, laplace = 0)
# train : 훈련 데이터를 포함하는 데이터 프레임 또는 행렬
# class : 훈련 데이터의 각 행에 대한 클래스를 갖는 팩터 벡터
# laplace : 라플라스 추정기를 제어하는 숫자(default = 0)

# 이 함수는 예측에 사용될 수 있는 나이브 베이즈 객체를 반환한다.


# 예측 :
# p <- predict(m, test, type = "class")
# m : naiveBayes() 함수에 의해 훈련된 모델
# test : 분류기를 구축하는 데 사용된 훈련 데이터와 같은 특징을 갖는 테스트 데이터를 포함하는 데이터 프레임 또는 행렬
# type : "class" 나 "raw" , 예측 결과가 가장 확률이 높은 클래스인지 원시 예측 확률인지를 지정

# 이 함수는 type 파라미터의 값에 따라 예측 클래스 값이나 원시 예측 확률의 벡터를 반환한다.


# 예제 : 
sms_classfier <- naiveBayes(sms_train, sms_type)
sms_prediction <- predict(sms_classfier, sms_test)


# sms_train 행렬에 있는 모델을 구축하려면 다음 명령을 이용한다.
sms_classfier <- naiveBayes(sms_train, sms_train_labels)

# 이제 sms_classfier 변수에는 예측을 위해 사용될 naiveBayes 분류기 객체가 들어있다.









### 4단계 : 모델 성능 평가

# SMS 분류기를 평가하려면 테스트 데이터의 낯선 메시지에 대해 예측을 테스트해볼 필요가 있다.

sms_test_pred <- predict(sms_classfier, sms_test)

# 예측을 실제 값과 비교하고자 이전에 사용했던 gmodels 모델 패키지에 있는 CrossTable() 함수를 사용한다.
# dnn 파라미터로 행과 열을 다음 코드처럼 다시 레이블을 지정할 것이다.
require(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = F, prop.c = F,
           prop.r = F, dnn = c('predicted', 'actual'))


# 표를 보면 1,390개의 SMS 메시지 중 6 + 30 = 36개(2.6%)만 부정확하게 분류된 것으로 확인된다.
# 오류는 햄 메시지 1,207개 중 스팸으로 잘못 분류된 6개이며, 183개의 스팸 메시지에서 30개가 햄으로 부정확하게 레이블됐다.
# 프로젝트에 투입한 노력이 적다는 것을 감안하면 이 수준의 성능은 매우 인상적인 것으로 보인다.

# 이 사례 연구는 나이브 베이즈가 텍스트 분류에 자주 사용되는 이유를 대표적으로 보여준다.
# 필터 때문에 중요한 문자 메시지를 놓칠 수 있기 때문에 더 나은 성능을 위해 모델을 약간 변경할 수 있는지를 확인할 수 있도록 조사해야만 한다.









### 5단계 : 모델 성능 개선

# 모델을 훈련할 때 라플라스 추정량을 설정하지 않았다는 것을 알았을 것이다
# 따라서 스팸 메시지에 나타나지 않거나 햄 메시지에 나타나지 않는 단어는 분류 과정에서 논쟁의 여지가 없는 확정적 결과를 출력해버릴 소지가 있다.

# 이전과 같이 나이브 베이즈 모델을 구축하지만, 이번에는 laplace = 1로 설정한다.
sms_classfier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classfier2, sms_test)

CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = F, prop.c = F, prop.r = F,
           dnn = c('predicted', 'actual'))


# 라플라스 추정량을 더함으로써 FP의 개수가 6개에서 5개로 줄어들고, FN의 개수가 30개에서 28개로 줄었다.
# 작은 변경처럼 보이겠지만, 모델의 정확도가 이미 매우 인상적이었다는 점을 감안하면 상당한 것이다.
# 스팸을 필터링하는 동안 지나치게 공격적인 것과 지나치게 수동적인 것 사이에서 균형을 유지하려면 모델을 너무 많이 조정하기 전에 신중해질 필요가 있다.
# 사용자는 햄 메시지가 너무 공격적으로 필터링 되는 것보다 스팸 메시지가 필터를 통과할 때 적게 빠지는 것을 선호할 것이다.

