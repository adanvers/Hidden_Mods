# Using Data Mining on Many Labs Data
library(haven)
library(caret)
library(xgboost)
library(ipred)
library(plyr)

# Reading in SPSS data
dat <- read_sav(file = "~/Dropbox/Data Mining Class/ManyLabs Project/Data/CleanedDataset.sav")

# creating variable for day study was conducted
dat$date <- as.Date(dat$last_update_date, format="%m/%d/%y")
dat$daysElapsed <- as.numeric(dat$date - min(dat$date))

# creating variable for time taken to complete study
dat$StartTime <- as.POSIXct(as.character(dat$creation_date),
                            format = "%Y-%m-%d %H:%M:%S")
dat$EndTime   <- as.POSIXct(as.character(dat$last_update_date),
                            format = "%m/%d/%y %H:%M")
dat$timeElapsed <- as.numeric(dat$EndTime - dat$StartTime)

# creating variable for time of day study was started
timeCh = strftime(dat$StartTime, format="%H:%M:%S")
time = as.POSIXct(timeCh, format="%H:%M:%S")
dat$timeOfday = as.numeric((time - min(time))/60)
# this variable can be interpreted as when the study was started, 
# relative to the earliest start time among all participants

### Examining Currency Priming

# to pre-process, all data must be converted to numeric
dat2 <- dat[,c("Sysjust", "MoneyGroup",
         "sex", "age", "race", "nativelang",
         "us_or_international", "moneypriorder", "separatedornot", 
         "daysElapsed", "timeOfday", "timeElapsed",
         "expgender", "exprace")]

dat2$separatedornot <- ifelse(dat2$separatedornot == "", "online", dat2$separatedornot)

# setting missing values
dat2 = as.data.frame(apply(dat2, 2, FUN=function(x) ifelse(x %in% c(".",""), NA, x)))

# making sure none of the variables are factors
dat2[,c("Sysjust", "MoneyGroup", "moneypriorder", "daysElapsed", "timeOfday", "timeElapsed")] <- 
  apply(dat2[,c("Sysjust", "MoneyGroup", "moneypriorder", "daysElapsed", "timeOfday", "timeElapsed")], 2, FUN=as.numeric)

# creating dummy variables with reference as largest category
dat2$nativelang = relevel(as.factor(dat2$nativelang), ref="english")
dat2$race = relevel(dat2$race, ref="6")
dat2$separatedornot = relevel(dat2$separatedornot, ref="online")
dat2$exprace = relevel(dat2$exprace, ref="6")
dat2$expfemale = relevel(dat2$expgender, ref="female")
dat2$female = relevel(dat2$sex, ref="f")

# remove all references to participant race with fewer than 20 cases
cutRace = names(which(sort(table(dat2$race)) < 20))
dat2$race2 = ifelse(as.character(dat2$race) %in% cutRace, NA, as.character(dat2$race))

datCat = dat2[,c("nativelang", "race2", "separatedornot", "exprace", "expfemale", "female")]

cats = dummyVars( ~., data=datCat, fullRank=T)
expandedCats = as.data.frame(predict(cats, newdata=datCat))

# saving a smaller data frame with categorical predictors and outcome
curr_dat = cbind(
  dat2[,c("Sysjust","MoneyGroup","moneypriorder","daysElapsed", "timeOfday", "timeElapsed")],
  expandedCats)

# split data into training and test sets
set.seed(365) # for reproducability
split = createDataPartition(curr_dat$MoneyGroup, times=1, p=0.8, list=F) # use caret's partitioning

curr_train = curr_dat[split,] # create training data set
curr_test = curr_dat[-split,] # create test data set

### IMPUTATION
# imputation of missing data using bagged trees
preProc <- preProcess(method="bagImpute", x = curr_train[, 2:length(curr_train)])
imputed <- predict(preProc, newdata=curr_train[, 2:length(curr_train)])

# train on three versions of data:
# 1. All predictors
# 2. Only Experimental Condition Included
# 3. Only Moderators Included

curr_train1 <- cbind(curr_train$Sysjust,imputed)
curr_train1 <- na.omit(curr_train1) # cut 10 cases with no DV measure

names(curr_train1)[1] <- "Sysjust"

curr_train2 = curr_train1[,c("Sysjust","MoneyGroup")]
curr_train3 = curr_train1[,-which(names(curr_train1)=="MoneyGroup")]

### Using boosted trees to predict outcome
ctl = trainControl(method="cv", number=10)

xgb1 = train(Sysjust ~ ., data=curr_train1, method="xgbTree", trControl=ctl)
xgb3 = train(Sysjust ~ ., data=curr_train3, method="xgbTree", trControl=ctl)

glm2 = train(Sysjust ~ ., data=curr_train2, method="glm", trControl=ctl)

# examining optimal parameters
xgb1$bestTune
xgb3$bestTune

# examining optimal fit
xgb1$results[which(xgb1$results$RMSE == min(xgb1$results$RMSE)),]
xgb3$results[which(xgb3$results$RMSE == min(xgb3$results$RMSE)),]

# examining variable importance
plot(varImp(xgb1))
plot(varImp(xgb3))

### out of sample prediction

# use imputation model to create test data
imputed_test <- predict(preProc, newdata=curr_test[, 2:length(curr_test)])
curr_test1 <- as.data.frame(cbind(curr_test$Sysjust, imputed_test))
names(curr_test1)[1] <- "Sysjust"
curr_test2 <- na.omit(curr_test1) # removing data from the test set with no value on the DV

# prediction
xgb1_pred = predict(xgb1, newdata=curr_test2)
sqrt(mean((xgb1_pred - na.omit(curr_test2$Sysjust))^2))
cor(xgb1_pred, na.omit(curr_test2$Sysjust))^2

xgb3_pred = predict(xgb3, newdata=curr_test2)
sqrt(mean((xgb3_pred - curr_test2$Sysjust)^2))
cor(xgb3_pred, curr_test2$Sysjust)^2

glm2_pred <- predict(glm2, newdata=curr_test2)
sqrt(mean((glm2_pred - curr_test2$Sysjust)^2))
cor(glm2_pred, curr_test2$Sysjust)^2

