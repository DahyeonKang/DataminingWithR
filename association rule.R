data(Titanic)
df_titan <- as.data.frame(Titanic)
df_titan



# install.packages('arules')
library(arules)
rules.all <- apriori(titanic)

options(digit=3)
inspect(rules.all)

rules <- apriori(titanic, control = list(verbose=F))
