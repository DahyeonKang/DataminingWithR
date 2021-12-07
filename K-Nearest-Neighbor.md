K-Nearest Neighbor
================
Dahyeon Kang
2021-12-08

### KNN이란

k-인접이웃분류() 모형은 새로운 데이터(설명변수)와 가장 유사한(거리가
가까운) k개의 과거 데이터(설명변수)의 결과(반응변수)를 이용해 다수결로
분류하는 모형이다. k값이 어떤 값으로 정해지는 것에 따라 분석 결과가
달라진다. 또한 반응변수가 범주형이면 classification, 반응변수가
연속형이면 regression으로 KNN이 사용된다.

KNN은 분류와 회귀에서 가까운 이웃에 대해 큰 가중(weight)를 부여할 수
있다. KNN은 데이터의 지역 구조에 민감하다는 단점을 가지고 있다.

### 예제를 통한 KNN 분석

{class} 패키지의 `knn()` 함수를 이용해 iris3 자료를 KNN 분류 분석해보자.
`knn(train, test, cl, k, ..)` :

-   `cl`, training set의 실제 분류 범주.

``` r
library(class)  # knn 함수를 위한 패키지
data(iris3)  # 50 x 4 x 3
head(iris3, 3)
```

    ## , , Setosa
    ## 
    ##      Sepal L. Sepal W. Petal L. Petal W.
    ## [1,]      5.1      3.5      1.4      0.2
    ## [2,]      4.9      3.0      1.4      0.2
    ## [3,]      4.7      3.2      1.3      0.2
    ## 
    ## , , Versicolor
    ## 
    ##      Sepal L. Sepal W. Petal L. Petal W.
    ## [1,]      7.0      3.2      4.7      1.4
    ## [2,]      6.4      3.2      4.5      1.5
    ## [3,]      6.9      3.1      4.9      1.5
    ## 
    ## , , Virginica
    ## 
    ##      Sepal L. Sepal W. Petal L. Petal W.
    ## [1,]      6.3      3.3      6.0      2.5
    ## [2,]      5.8      2.7      5.1      1.9
    ## [3,]      7.1      3.0      5.9      2.1

``` r
train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s", 25),rep("c", 25), rep("v", 25)))
knn(train, test, cl, k=3, prob=TRUE)
```

    ##  [1] s s s s s s s s s s s s s s s s s s s s s s s s s c c v c c c c c v c c c c
    ## [39] c c c c c c c c c c c c v c c v v v v v v v v v v c v v v v v v v v v v v
    ## attr(,"prob")
    ##  [1] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000
    ##  [8] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000
    ## [15] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000
    ## [22] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 0.6666667
    ## [29] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 0.6666667 1.0000000
    ## [36] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000
    ## [43] 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000
    ## [50] 1.0000000 1.0000000 0.6666667 0.7500000 1.0000000 1.0000000 1.0000000
    ## [57] 1.0000000 1.0000000 0.5000000 1.0000000 1.0000000 1.0000000 1.0000000
    ## [64] 0.6666667 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000
    ## [71] 1.0000000 0.6666667 1.0000000 1.0000000 0.6666667
    ## Levels: c s v

다음은 R {DMwR} 패키지의 `kNN()`함수를 이용해 KNN을 수행해보자. 제공하는
옵션 `norm=FALSE`는 정규화를 수행하지 않겠다는 옵션이고, default는
TRUE이다.

``` r
library(DMwR)
data(iris)
```

``` r
head(iris)
```

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
    ## 1          5.1         3.5          1.4         0.2  setosa
    ## 2          4.9         3.0          1.4         0.2  setosa
    ## 3          4.7         3.2          1.3         0.2  setosa
    ## 4          4.6         3.1          1.5         0.2  setosa
    ## 5          5.0         3.6          1.4         0.2  setosa
    ## 6          5.4         3.9          1.7         0.4  setosa

``` r
idxs <- sample(1:nrow(iris), as.integer(0.7*nrow(iris)))
trainset <- iris[idxs,]
testset <- iris[-idxs,]
model_3 <- kNN(Species ~ ., trainset, testset, norm=FALSE, k=3)
table(testset[,'Species'], model_3)
```

    ##             model_3
    ##              setosa versicolor virginica
    ##   setosa         11          0         0
    ##   versicolor      0         16         0
    ##   virginica       0          0        18

``` r
model_5_norm <- kNN(Species ~ ., trainset, testset, norm=TRUE, k=5)
table(testset[,'Species'], model_5_norm)
```

    ##             model_5_norm
    ##              setosa versicolor virginica
    ##   setosa         11          0         0
    ##   versicolor      0         16         0
    ##   virginica       0          0        18

``` r
model_6_norm <- kNN(Species ~ ., trainset, testset, norm=TRUE, k=6)
table(testset[,'Species'], model_6_norm)
```

    ##             model_6_norm
    ##              setosa versicolor virginica
    ##   setosa         11          0         0
    ##   versicolor      0         16         0
    ##   virginica       0          0        18

``` r
model_8 <- kNN(Species ~ ., trainset, testset, norm=FALSE, k=8)
table(testset[,'Species'], model_8)
```

    ##             model_8
    ##              setosa versicolor virginica
    ##   setosa         11          0         0
    ##   versicolor      0         16         0
    ##   virginica       0          0        18

사실 iris 데이터는 단위가 다른 변수가 있는 게 아니기 때문에 정규화를
시키지 않아도 된다.

다음은 R {kknn} 패키지의 `kknn()`함수를 이용해 KNN을 수행해보자. 먼저
train, test set으로 나누기 위해 비복원(`raplace=FALSE`)으로 추출.  
`kknn(formula, train, test, distance, nernel, ..)`:

-   `distance`, 인접한 이웃을 구하기 위한 거리의 모수 지정. default는
    민코우스키 거리, distance=2는 유클리드거리에 해당.
-   `kernel`, 이웃의 weight를 부여하는 방법 지정. “rectangular”,
    “trainagular” 등이 있는데 rectangular은 가중을 고려하지 않은 표준
    KNN이다.

``` r
library(kknn)
```

    ## Warning: package 'kknn' was built under R version 4.0.5

``` r
data(iris)
m <- dim(iris)[1]  # 150
val <- sample(1:m, size=round(m/3), replace=FALSE, prob=rep(1/m, m))
iris.learn <- iris[-val,]
iris.valid <- iris[val,]
iris.kknn <- kknn(Species ~ ., iris.learn, iris.valid, distance=1, kernel='triangular')
summary(iris.kknn)
```

    ## 
    ## Call:
    ## kknn(formula = Species ~ ., train = iris.learn, test = iris.valid,     distance = 1, kernel = "triangular")
    ## 
    ## Response: "nominal"
    ##           fit prob.setosa prob.versicolor prob.virginica
    ## 1      setosa           1      0.00000000     0.00000000
    ## 2      setosa           1      0.00000000     0.00000000
    ## 3   virginica           0      0.00000000     1.00000000
    ## 4  versicolor           0      1.00000000     0.00000000
    ## 5      setosa           1      0.00000000     0.00000000
    ## 6   virginica           0      0.30008950     0.69991050
    ## 7   virginica           0      0.00000000     1.00000000
    ## 8   virginica           0      0.00000000     1.00000000
    ## 9   virginica           0      0.00000000     1.00000000
    ## 10  virginica           0      0.19897037     0.80102963
    ## 11 versicolor           0      1.00000000     0.00000000
    ## 12  virginica           0      0.28907791     0.71092209
    ## 13     setosa           1      0.00000000     0.00000000
    ## 14 versicolor           0      0.96380648     0.03619352
    ## 15  virginica           0      0.00000000     1.00000000
    ## 16  virginica           0      0.00000000     1.00000000
    ## 17     setosa           1      0.00000000     0.00000000
    ## 18  virginica           0      0.21542752     0.78457248
    ## 19 versicolor           0      1.00000000     0.00000000
    ## 20  virginica           0      0.00000000     1.00000000
    ## 21 versicolor           0      1.00000000     0.00000000
    ## 22  virginica           0      0.08775331     0.91224669
    ## 23     setosa           1      0.00000000     0.00000000
    ## 24  virginica           0      0.00000000     1.00000000
    ## 25     setosa           1      0.00000000     0.00000000
    ## 26     setosa           1      0.00000000     0.00000000
    ## 27  virginica           0      0.00000000     1.00000000
    ## 28 versicolor           0      1.00000000     0.00000000
    ## 29     setosa           1      0.00000000     0.00000000
    ## 30     setosa           1      0.00000000     0.00000000
    ## 31 versicolor           0      0.75269021     0.24730979
    ## 32     setosa           1      0.00000000     0.00000000
    ## 33  virginica           0      0.00000000     1.00000000
    ## 34     setosa           1      0.00000000     0.00000000
    ## 35 versicolor           0      1.00000000     0.00000000
    ## 36     setosa           1      0.00000000     0.00000000
    ## 37 versicolor           0      1.00000000     0.00000000
    ## 38  virginica           0      0.00000000     1.00000000
    ## 39  virginica           0      0.00000000     1.00000000
    ## 40 versicolor           0      1.00000000     0.00000000
    ## 41     setosa           1      0.00000000     0.00000000
    ## 42     setosa           1      0.00000000     0.00000000
    ## 43     setosa           1      0.00000000     0.00000000
    ## 44     setosa           1      0.00000000     0.00000000
    ## 45  virginica           0      0.47771747     0.52228253
    ## 46 versicolor           0      1.00000000     0.00000000
    ## 47  virginica           0      0.13276820     0.86723180
    ## 48 versicolor           0      0.55746368     0.44253632
    ## 49 versicolor           0      1.00000000     0.00000000
    ## 50  virginica           0      0.00000000     1.00000000

``` r
fit <- fitted(iris.kknn)
# 적합
table(iris.valid$Species, fit)
```

    ##             fit
    ##              setosa versicolor virginica
    ##   setosa         17          0         0
    ##   versicolor      0         13         2
    ##   virginica       0          0        18

``` r
# 시각화
pcol <- as.character(as.numeric(iris.valid$Species))
pairs(iris.valid[1:4], pch=pcol, col=c('green3', 'red')[(iris.valid$Species != fit) + 1]) # green-정답(1), red-오답(2)
```

![](K-Nearest-Neighbor_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

다음은 R {FNN} 패키지를 사용해 검증용 데이터에 가장 가까운 k개의 자료를
구체적으로 구하는 2가지 방법을 제시한다.  
ex) 미 프로야구 선수 6명에 대해 두 시즌 간의 기록(lag1, lag2)이 다음
해의 득점(runs)에 미친 영향을 알아보기 위해 KNN 회귀를 수행한다.

``` r
df <- data.frame(name=c('McGwire,Mark', 'Bonds,Barry', 'Helton,Todd', 'Walker,Larry', 'Pujols,Albert', 'Pedroia,Dustin'),
                   lag1=c(100,90,75,89,95,70),
                   lag2=c(120,80,95,79,92,90),
                   runs=c(65,120,105,99,65,100))
df
```

    ##             name lag1 lag2 runs
    ## 1   McGwire,Mark  100  120   65
    ## 2    Bonds,Barry   90   80  120
    ## 3    Helton,Todd   75   95  105
    ## 4   Walker,Larry   89   79   99
    ## 5  Pujols,Albert   95   92   65
    ## 6 Pedroia,Dustin   70   90  100

``` r
library(kknn)
trainset <- df[df$name != 'Bonds,Barry', ]
testset <- df[df$name == 'Bonds,Barry', ]
model <- kknn(runs ~ lag1 + lag2, train=trainset, test=testset, k=2, distance=1)
fit <- fitted(model)
fit
```

    ## [1] 90.5

총 6명의 선수자료 중에 5명의 선수자료를 train으로, 1명의 선수자료를
test로 데이터를 나누었다. KNN 예측한 결과로 90.5를 얻었는데 이는
`model$fitted.values` 통해서도 알 수 있다.

``` r
names(model)
```

    ##  [1] "fitted.values" "CL"            "W"             "D"            
    ##  [5] "C"             "prob"          "response"      "distance"     
    ##  [9] "call"          "terms"

-   `CL`, k-근접이웃의 class 행렬  
-   `W`, k-근접이웃의 weight 행렬  
-   `D`, k-근접이웃의 거리 행렬  
-   `C`, k-근접이웃의 위치(indices)

``` r
model$fitted.values
```

    ## [1] 90.5

``` r
model$CL
```

    ##      [,1] [,2]
    ## [1,]   99   65

``` r
model$C
```

    ## [1] 3 4

``` r
model$W
```

    ##      [,1] [,2]
    ## [1,] 0.75 0.25

Bonds 선수와 가장 가까운 2명의 선수의 득점(runs), 위치, 가중치를 얻었다.
예측값(fitted.values) 90.5는 가중평균 (99 \* 3 + 65 \* 1)/4 = 90.5해서
구해진 값이다.  
하지만 인접값(CL) 99는 선수 1명(Walker)이, 65는 2명(McGwire, Pujols)의
선수가 가지고 있다. 65에 해당하는 정확한 선수를 구해보자.

``` r
model$C
```

    ## [1] 3 4

``` r
trainset[c(model$C), ]
```

    ##            name lag1 lag2 runs
    ## 4  Walker,Larry   89   79   99
    ## 5 Pujols,Albert   95   92   65

인덱스가 4, 5로 나오지만 이 인덱스를 무시하고, 위에서 순서대로 따지면
trainset에서 3, 4번째에 위치해 있다. 따라서 Waler와 Pujols가 Bonds의
인접 이웃이다.

``` r
trainset
```

    ##             name lag1 lag2 runs
    ## 1   McGwire,Mark  100  120   65
    ## 3    Helton,Todd   75   95  105
    ## 4   Walker,Larry   89   79   99
    ## 5  Pujols,Albert   95   92   65
    ## 6 Pedroia,Dustin   70   90  100

R 패키지 {FNN}의 `get.knnx()`를 통해 인접 이웃을 구해보자.

``` r
library(FNN)
```

    ## Warning: package 'FNN' was built under R version 4.0.5

    ## 
    ## Attaching package: 'FNN'

    ## The following objects are masked from 'package:class':
    ## 
    ##     knn, knn.cv

``` r
get.knnx(data=trainset[ , c('lag1', 'lag2')],
         query=testset[ , c('lag1', 'lag2')], k=2)
```

    ## $nn.index
    ##      [,1] [,2]
    ## [1,]    3    4
    ## 
    ## $nn.dist
    ##          [,1] [,2]
    ## [1,] 1.414214   13

nn.index가 인접이웃의 index가 3,4번이라고 알려주고, nn.dist가
인접이웃과의 default 거리인 Euclidean 거리 계산값을 보여준다.

``` r
trainset[c(3, 4), 'name']
```

    ## [1] "Walker,Larry"  "Pujols,Albert"

위 `trainset[c(model$C), ]` 코드와 마찬가지로 인접이웃을 알려준다.

### {caret}을 이용한 KNN 분석

R 패키지 {caret}을 통해서 KNN을 수행해보자.

#### (a)표본추출

`createDataPartition()`로 매우 편리하게 자료를 분할할 수 있다.  
\* `y`, y(반응변수)의 class 혹은 label.  
\* `p`, train 데이터에 사용할 전체 데이터에서의 비율.  
\* `list`, 분할한 결과를 리스트로 반환할지 여부. FALSE이면 matrix
반환.  
`prop.table()`은 matrix를 proportion 테이블로 변환시켜주는 함수.

``` r
library(ISLR)
```

    ## Warning: package 'ISLR' was built under R version 4.0.5

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.0.5

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.0.5

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:kknn':
    ## 
    ##     contr.dummy

``` r
set.seed(100)
train.idx <- createDataPartition(y=Smarket$Direction, p=0.75, list=FALSE)
trainset <- Smarket[train.idx, ]
testset <- Smarket[-train.idx, ]
prop.table(table(trainset$Direction)) * 100
```

    ## 
    ##     Down       Up 
    ## 48.18763 51.81237

``` r
prop.table(table(testset$Direction)) * 100
```

    ## 
    ##     Down       Up 
    ## 48.07692 51.92308

``` r
prop.table(table(Smarket$Direction)) * 100
```

    ## 
    ##  Down    Up 
    ## 48.16 51.84

#### (b)전처리

KNN 분류를 수행하기 위해 변수의 정규화 혹은 척도화가 필효하다. R 패키지
{caret}의 `preProcess()`함수를 통해 중심화, 척도화 전처리를 해준다.

``` r
train.x <- trainset[ , names(trainset) != 'Direction']  # 반응변수(y)를 제외
(pre.values <- preProcess(x=train.x, method=c('center', 'scale')))
```

    ## Created from 938 samples and 8 variables
    ## 
    ## Pre-processing:
    ##   - centered (8)
    ##   - ignored (0)
    ##   - scaled (8)

#### (C)훈련과 훈련 조절

cross validation에 기초해 적합한 결과, 인접 이웃의 크기 k가 29일 때
모델이 가장 성능이 좋다고 나온다.

``` r
set.seed(200)
control <- trainControl(method='repeatedcv', repeats=3)
knn.fit <- train(Direction ~ ., data=trainset, method='knn',
                 trControl=control,
                 preProcess=c('center', 'scale'), tuneLength=20)
knn.fit
```

    ## k-Nearest Neighbors 
    ## 
    ## 938 samples
    ##   8 predictor
    ##   2 classes: 'Down', 'Up' 
    ## 
    ## Pre-processing: centered (8), scaled (8) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 844, 844, 843, 844, 845, 845, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   Accuracy   Kappa    
    ##    5  0.8837625  0.7668667
    ##    7  0.8823439  0.7641736
    ##    9  0.8798806  0.7591151
    ##   11  0.8930660  0.7855027
    ##   13  0.8923603  0.7840046
    ##   15  0.8983926  0.7959911
    ##   17  0.9008712  0.8008697
    ##   19  0.8998034  0.7987302
    ##   21  0.9015651  0.8021944
    ##   23  0.9015574  0.8021663
    ##   25  0.9072465  0.8135966
    ##   27  0.9072542  0.8135649
    ##   29  0.9111626  0.8214699
    ##   31  0.9090274  0.8171458
    ##   33  0.9058319  0.8107612
    ##   35  0.9097479  0.8186335
    ##   37  0.9076353  0.8143704
    ##   39  0.9065752  0.8122363
    ##   41  0.9080164  0.8150995
    ##   43  0.9090842  0.8172527
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 29.

``` r
plot(knn.fit)
```

![](K-Nearest-Neighbor_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
knn.predict <- predict(knn.fit, newdata=testset)
confusionMatrix(knn.predict, testset$Direction)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Down  Up
    ##       Down  124   8
    ##       Up     26 154
    ##                                           
    ##                Accuracy : 0.891           
    ##                  95% CI : (0.8511, 0.9233)
    ##     No Information Rate : 0.5192          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7808          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.003551        
    ##                                           
    ##             Sensitivity : 0.8267          
    ##             Specificity : 0.9506          
    ##          Pos Pred Value : 0.9394          
    ##          Neg Pred Value : 0.8556          
    ##              Prevalence : 0.4808          
    ##          Detection Rate : 0.3974          
    ##    Detection Prevalence : 0.4231          
    ##       Balanced Accuracy : 0.8886          
    ##                                           
    ##        'Positive' Class : Down            
    ## 
