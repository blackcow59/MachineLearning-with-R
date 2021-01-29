# R 데이터 구조

# 머신러닝에서 가장 빈번하게 사용되는 R 데이터 구조에는 벡터(vector), 팩터(factor), 배열(array), 행렬(matrix), 데이터 프레임(data frame) 이 있다.


### 벡터(vector)
# 벡터는 가장 기본적인 R 데이터 구조로 항목이라고 하는 값의 순서 집합을 저장한다.
# 벡터 항목은 개수 제한이 없지만, 모두 같은 형식이어야 한다.
# 벡터 v의 형식을 판단하려면 typeof(v)명령을 사용하면 된다.

# 벡터타입
# interger(소수 자리가 없는 숫자)
# double(소수 자리가 있는 숫자)
# character(텍스트 데이터)
# logical(TRUE or FALSE)
# NULL(값이 존재하지 않음)
# NA(결측치)


subject_name <- c("John Doe", "Jane Doe", "Steve Graves")
temperature <- c(98.1, 98.6, 101.4)
flu_status <- c(F, F, T)

temperature[2]
temperature[2:3]
temperature[-2]
temperature[c(T, T, F)]





### 펙터(factor)
# 펙터는 오직 범주변수나 서열변수만을 나타내고자 사용되는 특별한 종류의 벡터이다

gender <- factor(c("MALE", "FEMALE", "MALE"))
gender

blood <- factor(c("O", "AB", "A"),
                levels = c("A", "B", "AB", "O"))
blood


symptoms <- factor(c("SEVERE", "MILD", "MODERATE"),
                   levels = c("MILD", "MODERATE", "SEVERE"), 
                   ordered = T)
symptoms
symptoms > "MODERATE"





### 리스트(list)
# 리스트는 항목의 순서 집합을 젖아하는 데 사용된다는 점엣허 벡터와 매우 유사한 데이터 구조이다.
# 벡터와 달리 수집될 항목이 다른 R데이터 타입이어도 된다.

subject_name[1]
temperature[1]
flu_status[1]
gender[1]
blood[1]
symptoms[1]


# 리스트를 구성할 때 열의 각 구성 요소에는 이름이 주어져야 한다.
# 반드시 필요한 것은 아니지만 이를 통해 나중에 리스트의 값에 접근할 때 숫자로 된 위치가 아닌 이름으로 접근할 수 있다.

subject <- list(fullname = subject_name,
                temperature = temperature,
                flu_status = flu_status,
                gender = gender,
                blood = blood,
                symptoms = symptoms)
subject
subject[2]
subject[[2]]
subject$temperature
subject[c("temperature", "flu_status")]





### 데이터 프레임(data frame)
# 데이터의 행과 열을 갖고 있기 떄문에 시프레드시트나 데이터베이스와 유사한 구조다
# 정확히 동일한 개수의 값을 갖는 벡터나 펙터의 리스트다.

pt_data <- data.frame(subject_name, temperature, flu_status, gender, blood, symptoms)
pt_data
str(pt_data)

# 1차원 벡터나 리스트와 비교해보면 데이터 프레임은 2차원이고 행렬 형식으로 나타난다.
# 데이터 프레임의 열은 특징이나 속성이고 행은 예시이다.

pt_data$subject_name
pt_data[c("temperature", "flu_status")]
pt_data[2:3]

# 데이터 프레임은 2차원이기 떄문에 추출을 원하는 행과 열을 모두 명시해야 한다. [rows, columns]

pt_data[1,2]
pt_data[c(1,3), c(2,4)]
pt_data[, 1]
pt_data[1,]
pt_data[ , ]
pt_data[c(1,3), c("temperature", "gender")]
pt_data[-2, c(-1, -3, -5, -6)]

pt_data$temp_c <- (pt_data$temperature - 32) *(5 / 9)
pt_data[c("temperature", "temp_c")]





### 행렬(matrix)과 배열(array)
# 행렬은 데이터의 행과 열을 갖는 2차원 표를 나타내는 데이터 구조다.
# 행렬은 벡터처럼 동일한 형식의 데이터만 가질 수 있으며, 대개 수학 연산에 가장 자주 사용되므로 일반적으로 수치 데이터만 저장한다.

m <- matrix(c(1,2,3,4 ), nrow = 2)
m
m <- matrix(c(1,2,3,4), ncol = 2)
m
m <- matrix(c(1,2,3,4,5,6), nrow = 2)
m
m <- matrix(c(1,2,3,4,5,6), ncol = 2)
m

m[1,1]
m[3,2]
m[1,]
m[,1]












# R을 이용한 데이터 관리
# R로 데이터를 가져오고, R에서 데이터를 내보내는 기초적인 기능을 살펴본다.



### 데이터 구조 저장, 로드, 제거

# save(객체, file = "파일이름")
save(x, y, z, file = "mydata.RData")

# load("파일이름")
load("mydata.RData")

# R 세션을 서둘러 정리해야 한다면 save.image() 명령으로 전체 세션을 .RData 파일에 저장한다.
# R이 재시작될 때 R은 디폴트로 이 파일을 찾아 세션을 떠날 때처럼 세션을 재생성한다.


# ls() : 현재 메모리에 있는 데티어 구조의 목록 벡터를 반환한다.
ls()

# rm(객체) : 객체를 제거한다
rm(m, subject)
rm(list = ls()) # R 세션의 전체 객체를 해제






### CSV 파일에서 데이터 임포트와 저장

# read.csv("path/to/data.csv", header = T/F, ...) : csv파일 데이터 불러오기

# write.csv(data, file = "filename.csv", row.names = T/F(csv파일에 행 이름 출력 여부), ...) : 데이터 프레임을 csv 파일에 저장하기










# 데이터 탐색과 이해
# 데이터 특징과 예시를 탐색하고 데이터를 고유하게 만들어줄 특성을 알아본다.
usedcars <- read.csv("data/usedcars.csv")
usedcars





### 데이터 구조 탐색
str(usedcars)



### 수치 변수 탐색
summary(usedcars$year)

summary(usedcars[c("price", "mileage")])

# 중심 경향 측정 : 평균(mean)과 중앙값(median)
# 퍼짐 측정 : 사분위수와 다섯 숫자 요약(quantile) : 최솟값(min), 1사분위수(Q1), 중앙값(median), 3사분위수(Q3), 최댓값(max)
# 범위(range) : 최솟값과 최댓값 사이의 폭
range(usedcars$price)
diff(range(usedcars$price))

# 사분위 범위(IQR)
IQR(usedcars$price)

quantile(usedcars$price)
quantile(usedcars$price, probs = c(0.01, 0.99))
quantile(usedcars$price, seq(from = 0, to = 1, by = 0.2))


# 수치 변수 시각화 : 상자 그림(boxplot)
boxplot(usedcars$price, main = "Boxplot of Used Car Prices", ylab = "Price ($)")
boxplot(usedcars$mileage, main = "Boxplot of Used car Mileage", ylab = "Odometer (mi.)")



# 수치 변수 시각화 : 히스토그램(hist)
hist(usedcars$price, main = "Histogram of Used Car Prices", xlab = "Price ($)")
hist(usedcars$mileage, main = "Histogram of Used Car Mileage", xlab = "Odometer (mi.)")
hist(usedcars$mileage, main = "Histogram of Used Car Mileage", xlab = "Odometer (mi.)", breaks = 10)
hist(usedcars$mileage, main = "Histogram of Used Car Mileage", xlab = "Odometer (mi.)", breaks = c(0, 50000, 100000, 150000, 200000))



# 수치 데이터의 이해 : 균등 분포와 정규 분포

# 퍼짐 측정 : 분산(var)과 표준편차(sd)
var(usedcars$price)
sd(usedcars$price)
var(usedcars$mileage)
sd(usedcars$mileage)





### 범주 변수 탐색

table(usedcars$year)
table(usedcars$model)
table(usedcars$color)

model_table <- table(usedcars$model)
prop.table(model_table)
round(prop.table(model_table), digits = 1)

color_table <- table(usedcars$color)
color_pct <- prop.table(color_table) *100
round(color_pct, digits = 1)


# 중심 경향 측정 : 최빈값
# 다른 범주와 비교해서 최반값을 생각하는 것이 가장 좋다.
# 하나의 범주나 몇 개의 범주가 다른 모든 범주를 압도한다면 어떨까?
# 이런 식으로 최빈값을 생각하면 어떤 특정 값이 다른 값보다 더 흔한 이유가 무엇인지에 대한 의문을 검정 가능한 가설로 설정할 수 있다.


# 변수 간의 관계 탐색
# price와 mileage 데이터를 볼 때 이 데이터는 경차만 포함한다고 봐야 하는가? 아니면 주행거리가 긴 고급 차라고 봐야 하는가?
# model과 color 데이터 간의 관계는 조사중인 차의 유형에 대해 어떤 통찰력을 제공해 주는가?


# 관계 시각화 : 산포도(산점도)
# 산포도(산점도) : 수치 특징 사이의 이변량 관계를 시각화하는 다이어그램
plot(price ~ mileage, data = usedcars, 
     main = "Scatterplot of Price vs. Mileage",
     xlab = "Used Car Odometer (mi.)",
     ylab = "Used Car Price ($)")


# 관계 관찰 : 이원 교차표
# install.packages("gmodels")
require(gmodels)
usedcars$conservative <- usedcars$color %in% c("Black", "Gray", "Silver", "White")
# %in% : 연산자 좌변의 벡터 값이 오른쪽 벡터에 존재하는지 여부에 따라 TRUE 나 FALSE를 반환한다.

table(usedcars$conservative)

CrossTable(x = usedcars$model, y = usedcars$conservative)
CrossTable(x = usedcars$model, y = usedcars$conservative, chisq = T)
