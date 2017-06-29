# Using Data Mining on Many Labs Data
library(haven)
library(caret)
library(xgboost)

# Reading in SPSS data
dat <- read_sav(file = "~/Dropbox/Data Mining Class/ManyLabs Project/Data/CleanedDataset.sav")

# creating variable for day study was conducted
dat$date <- as.Date(dat$last_update_date, format="%m/%d/%y")
dat$daysElapsed <- dat$date - min(dat$date)

# creating variable for time study was completed
dat$timeOfdayC = substr(as.character(dat$last_update_date), start=9, stop=13)
time = strptime(dat$timeOfdayC, format="%H:%M")
dat$timeOfday = (time - time[1])/60

# creating variable for time taken to complete study
dat$StartTime <- as.POSIXct(as.character(dat$creation_date),
                            format = "%Y-%m-%d %H:%M:%S")
dat$EndTime   <- as.POSIXct(as.character(dat$last_update_date),
                            format = "%m/%d/%y %H:%M")
dat$timeElapsed = (dat$EndTime - dat$StartTime)

sort(table(dat$exprace))
sort(table(dat$race))
sort(table(dat$nativelang))

### Examining Currency Priming

# saving a smaller data frame with predictors and outcome
curr_dat = dat[,c("Sysjust", "MoneyGroup",
                  "sex", "age", "race", "nativelang",
                  "us_or_international", "moneypriorder", "separatedornot", 
                  "daysElapsed", "timeOfday", "timeElapsed",
                  "expgender", "exprace")]

# listwise deletion
curr_dat_noNA = na.omit(curr_dat)

# split data into training and test sets
set.seed(365) # for reproducability
split = createDataPartition(curr_dat_noNA$MoneyGroup, times=1, p=0.8, list=F) # use caret's partitioning

curr_train = curr_dat_noNA[split,] # create training data set
curr_test = curr_dat_noNA[-split,] # create test data set

# train on three versions of data:
# 1. All predictors
# 2. Only Experimental Condition Included
# 3. Only Moderators Included

curr_train2 = curr_train[,c("Sysjust","MoneyGroup")]
curr_train3 = curr_train[,c("Sysjust",
                            "sex", "age", "race", "nativelang",
                            "us_or_international", "moneypriorder", "separatedornot", 
                            "daysElapsed", "timeElapsed",
                            "expgender", "exprace")]

### Using boosted trees to predict outcome
ctl = trainControl(method="cv", number=10)

xgb1 = train(Sysjust ~ ., data=curr_train, method="xgbTree", trControl=ctl)
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

# checking that all levels are present
catmods = c("sex","race", "nativelang",
            "us_or_international","separatedornot","expgender", "exprace")
lvlsgood = matrix(NA, 1, length(catmods))

for (i in 1:length(catmods)) {
   lvlsgood[i] = ifelse(length(names(table(curr_train[,catmods[i]]))) >= length(names(table(curr_train[,catmods[i]]))), 1, 0)
}

lvlsgood

# prediction
xgb1_pred = predict(xgb1, newdata=curr_test)
sqrt(mean((xgb1_pred - curr_test$Sysjust)^2))
cor(xgb1_pred, curr_test$Sysjust)^2

xgb3_pred = predict(xgb3, newdata=curr_test)
sqrt(mean((xgb3_pred - curr_test$Sysjust)^2))
cor(xgb3_pred, curr_test$Sysjust)^2
