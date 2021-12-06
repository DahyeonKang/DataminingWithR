K-Nearest Neighbor
================
Dahyeon Kang
2021-12-06

### KNN이란

k-인접이웃분류() 모형은 새로운 데이터(설명변수)와 가장 유사한(거리가
가까운) k개의 과거 데이터(설명변수)의 결과(반응변수)를 이용해 다수결로
분류하는 모형이다. k값이 어떤 값으로 정해지는 것에 따라 분석 결과가
달라진다. 또한 반응변수가 범주형이면 classification, 반응변수가
연속형이면 regression으로 KNN이 사용된다.

KNN은 분류와 회귀에서 가까운 이웃에 대해 큰 가중(weight)를 부여할 수
있다. KNN은 데이터의 지역 구조에 민감하다는 단점을 가지고 있다.

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
    ##   setosa         15          0         0
    ##   versicolor      0         14         1
    ##   virginica       0          2        13

``` r
model_5_norm <- kNN(Species ~ ., trainset, testset, norm=TRUE, k=5)
table(testset[,'Species'], model_5_norm)
```

    ##             model_5_norm
    ##              setosa versicolor virginica
    ##   setosa         15          0         0
    ##   versicolor      0         14         1
    ##   virginica       0          2        13

``` r
model_6_norm <- kNN(Species ~ ., trainset, testset, norm=TRUE, k=6)
table(testset[,'Species'], model_6_norm)
```

    ##             model_6_norm
    ##              setosa versicolor virginica
    ##   setosa         15          0         0
    ##   versicolor      0         15         0
    ##   virginica       0          3        12

``` r
model_8 <- kNN(Species ~ ., trainset, testset, norm=FALSE, k=8)
table(testset[,'Species'], model_8)
```

    ##             model_8
    ##              setosa versicolor virginica
    ##   setosa         15          0         0
    ##   versicolor      0         14         1
    ##   virginica       0          0        15

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
    ## 1  versicolor           0     1.000000000    0.000000000
    ## 2  versicolor           0     0.954786745    0.045213255
    ## 3   virginica           0     0.312442084    0.687557916
    ## 4      setosa           1     0.000000000    0.000000000
    ## 5      setosa           1     0.000000000    0.000000000
    ## 6      setosa           1     0.000000000    0.000000000
    ## 7   virginica           0     0.158875791    0.841124209
    ## 8  versicolor           0     0.772504193    0.227495807
    ## 9   virginica           0     0.094948125    0.905051875
    ## 10  virginica           0     0.003678874    0.996321126
    ## 11 versicolor           0     0.682085046    0.317914954
    ## 12 versicolor           0     1.000000000    0.000000000
    ## 13     setosa           1     0.000000000    0.000000000
    ## 14 versicolor           0     1.000000000    0.000000000
    ## 15  virginica           0     0.000000000    1.000000000
    ## 16  virginica           0     0.163262999    0.836737001
    ## 17     setosa           1     0.000000000    0.000000000
    ## 18 versicolor           0     0.895472823    0.104527177
    ## 19  virginica           0     0.257363876    0.742636124
    ## 20     setosa           1     0.000000000    0.000000000
    ## 21     setosa           1     0.000000000    0.000000000
    ## 22     setosa           1     0.000000000    0.000000000
    ## 23 versicolor           0     1.000000000    0.000000000
    ## 24  virginica           0     0.000000000    1.000000000
    ## 25     setosa           1     0.000000000    0.000000000
    ## 26 versicolor           0     1.000000000    0.000000000
    ## 27     setosa           1     0.000000000    0.000000000
    ## 28 versicolor           0     1.000000000    0.000000000
    ## 29 versicolor           0     1.000000000    0.000000000
    ## 30  virginica           0     0.000000000    1.000000000
    ## 31  virginica           0     0.000000000    1.000000000
    ## 32 versicolor           0     0.780239021    0.219760979
    ## 33  virginica           0     0.000000000    1.000000000
    ## 34 versicolor           0     1.000000000    0.000000000
    ## 35 versicolor           0     1.000000000    0.000000000
    ## 36 versicolor           0     0.996321061    0.003678939
    ## 37 versicolor           0     1.000000000    0.000000000
    ## 38  virginica           0     0.000000000    1.000000000
    ## 39     setosa           1     0.000000000    0.000000000
    ## 40 versicolor           0     1.000000000    0.000000000
    ## 41  virginica           0     0.344006240    0.655993760
    ## 42 versicolor           0     0.907502741    0.092497259
    ## 43     setosa           1     0.000000000    0.000000000
    ## 44  virginica           0     0.000000000    1.000000000
    ## 45     setosa           1     0.000000000    0.000000000
    ## 46     setosa           1     0.000000000    0.000000000
    ## 47     setosa           1     0.000000000    0.000000000
    ## 48  virginica           0     0.000000000    1.000000000
    ## 49  virginica           0     0.000000000    1.000000000
    ## 50  virginica           0     0.499660642    0.500339358

``` r
fit <- fitted(iris.kknn)
# 적합
table(iris.valid$Species, fit)
```

    ##             fit
    ##              setosa versicolor virginica
    ##   setosa         15          0         0
    ##   versicolor      0         16         0
    ##   virginica       0          2        17

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
trainset에서 3, 4번쨰에 위치해 있다. 따라서 Waler와 Pujols가 Bonds의
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
