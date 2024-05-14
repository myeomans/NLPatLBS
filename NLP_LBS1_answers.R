#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#              Assignment 1 Answers                 #
#                                                   #
#                                                   #
#####################################################

###############################################################
###############################################################
# Part 1 - Yelp Data
###############################################################
###############################################################

# Review data
review_dat<-readRDS("data/rev_med.RDS")

names(review_dat)

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(review_dat),round(nrow(review_dat)/2))

# create our prediction variables
dfm_rev<-TASSL_dfm(review_dat$text,ngrams=1)

###############################################################
# THE GENDER MODEL
###############################################################

trainX<-dfm_rev[train_split,]

trainY<-review_dat$male[train_split]

testX<-dfm_rev[-train_split,]

testY<-review_dat$male[-train_split]

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# split the predictions in two, using the median

# note - the target Y variable is 1/0 so we have to convert to 1/0, not 2/1
test_predict_binary=ifelse(test_predict>median(test_predict),
                           1,
                           0) 

# calculate accuracy

round(100*mean(test_predict_binary==testY),3)


pROC::roc(testY,test_predict,ci=T)

plot(pROC::roc(testY,test_predict,ci=T))

#####################
# Build a plot
#####################

# extract coefficients
plotCoefs<-lasso_model %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(trainX),
                       freq=colMeans(trainX))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_x_continuous(#limits = c(-.2,.25),
    breaks = seq(-.2,.2,.1)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Gender Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

###############################################################
###############################################################
# Part 2 - Glassdoor Data
###############################################################
###############################################################


glassdoor<-readRDS("data/glassdoorReviews.RDS")

# Note - there are two different text boxes!! "pros" and "cons"
glassdoor <- glassdoor%>%
  mutate(pros_wordcount=str_count(pros,"[[:alpha:]]+"),
         cons_wordcount=str_count(cons,"[[:alpha:]]+"))

gd_small<-glassdoor %>%
  filter(pros_wordcount>5 & cons_wordcount>5)

# before we randomize, use set.seed() to all get the same split
set.seed(02138)

# grab the first 40,000 rows after randomizing
gd_small<-gd_small %>%
  arrange(sample(1:n())) %>%
  slice(1:40000)

##############################################################
# split into train and test
train_split=sample(1:nrow(gd_small),20000)

gd_train<-gd_small%>%
  slice(train_split)

gd_test<-gd_small%>%
  slice(-train_split)

##############################################################
# Let's just look at amazon for now

gd_amazon_train<-gd_train %>%
  filter(company=="amazon")

gd_amazon_test<-gd_test %>%
  filter(company=="amazon")

# create our prediction variables from the pros text
dfm_amazon_train_pros<-TASSL_dfm(gd_amazon_train$pros,ngrams=1:2) %>%
  convert(to="matrix")

amazon_train_Y<-gd_amazon_train %>%
  pull(overall)

# Put training data into LASSO model

amazon_model_pros<-cv.glmnet(x=dfm_amazon_train_pros,
                             y=amazon_train_Y)

# First, let's test the model on the pros text from amazon
dfm_amazon_test_pros<-TASSL_dfm(gd_amazon_test$pros,
                                ngrams=1:2,
                                min.prop = 0) %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

amazon_test_Y<-gd_amazon_test %>%
  pull(overall)

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_pros)[,1]

# estimate accuracy - use kendall's tau
pros_acc_p<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

############################################

# Let's apply the same model to the cons text

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_cons)[,1]

# estimate accuracy
cons_acc_p<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

#################################################
# Question 3
#################################################

# create our prediction variables from the cons text
dfm_amazon_train_cons<-TASSL_dfm(gd_amazon_train$cons,ngrams=1:2) %>%
  convert(to="matrix")

amazon_model_cons<-cv.glmnet(x=dfm_amazon_train_cons,
                             y=amazon_train_Y)

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_cons)[,1]
# estimate accuracy
cons_acc_c<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

dfm_amazon_test_pros<-TASSL_dfm(gd_amazon_test$pros,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_pros)[,1]
# estimate accuracy
pros_acc_c<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

#############################################
# Combine accuracy estimates for a plot
#############################################
bind_rows(pros_acc_p %>%
            mutate(test="Pros",
                   train="Pros Model"),
          cons_acc_p %>%
            mutate(test="Cons",
                   train="Pros Model"),
          pros_acc_c %>%
            mutate(test="Pros",
                   train="Cons Model"),
          cons_acc_c %>%
            mutate(test="Cons",
                   train="Cons Model")) %>%
  ggplot(aes(x=test,color=test,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  facet_wrap(~train) +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        strip.text = element_text(size=24),
        panel.grid=element_blank(),
        strip.background = element_rect(fill="white"),
        legend.position="none")


#################################################
# Question 4
#################################################

# estimate accuracy
pros_acc_wdct<-kendall_acc(gd_amazon_test$pros_wordcount,amazon_test_Y)

cons_acc_wdct<-kendall_acc(gd_amazon_test$cons_wordcount,amazon_test_Y)


bind_rows(pros_acc_p %>%
            mutate(test="Pros",
                   train="Pros Model"),
          cons_acc_p %>%
            mutate(test="Cons",
                   train="Pros Model"),
          pros_acc_c %>%
            mutate(test="Pros",
                   train="Cons Model"),
          cons_acc_c %>%
            mutate(test="Cons",
                   train="Cons Model"),
          pros_acc_wdct %>%
            mutate(test="Pros",
                   train="Word Count"),
          cons_acc_wdct %>%
            mutate(test="Cons",
                   train="Word Count")) %>%
  ggplot(aes(x=test,color=test,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  facet_wrap(~train) +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        strip.text = element_text(size=24),
        panel.grid=element_blank(),
        strip.background = element_rect(fill="white"),
        legend.position="none")



#################################################
# Question 5
#################################################

# Pros Plot

# extract coefficients
plotCoefs<-amazon_model_pros %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_pros),
                       freq=colMeans(dfm_amazon_train_pros))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Pros Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

#################
# Cons Plot
#################

# extract coefficients
plotCoefs<-amazon_model_cons %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_cons),
                       freq=colMeans(dfm_amazon_train_cons))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Cons Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#################################################
# Question 6
#################################################


# store predictions in data, calculate accuracy
gd_amazon_test<-gd_amazon_test %>%
  mutate(prediction=amazon_test_predict_cons,
         error=abs(overall-prediction),
         bias=overall-prediction)

close_high<-gd_amazon_test %>%
  filter(overall==5 & error<.5) %>%
  select(cons,overall,prediction)

close_low<-gd_amazon_test %>%
  filter(overall==1 & error<.5) %>%
  select(cons,overall,prediction)

close_high
close_high %>%
  slice(1:2) %>%
  pull(cons)

close_low
close_low %>%
  slice(1:2) %>%
  pull(cons)


#################################################
# Question 7
#################################################

# Error analysis - find biggest misses

gd_amazon_test %>%
  ggplot(aes(x=prediction)) +
  geom_histogram()

gd_amazon_test %>%
  ggplot(aes(x=overall)) +
  geom_histogram()

miss_high<-gd_amazon_test %>%
  arrange(bias) %>%
  slice(1:10) %>%
  select(cons,overall,prediction)

miss_low<-gd_amazon_test %>%
  arrange(-bias) %>%
  filter(overall==5) %>%
  slice(1:10) %>%
  select(cons,overall,prediction)

miss_low
miss_low%>%
  slice(1:2) %>%
  pull(cons)

miss_high
miss_high%>%
  slice(3) %>%
  pull(cons)


#################################################
# Question 8
#################################################


gd_microsoft_train<-gd_train %>%
  filter(company=="microsoft")

gd_microsoft_test<-gd_test %>%
  filter(company=="microsoft")

microsoft_train_Y<-gd_microsoft_train %>%
  pull(overall)

microsoft_test_Y<-gd_microsoft_test %>%
  pull(overall)


# create our prediction variables from the cons text
dfm_microsoft_train_cons<-TASSL_dfm(gd_microsoft_train$cons,ngrams=1:2) %>%
  convert(to="matrix")

microsoft_model_cons<-cv.glmnet(x=dfm_microsoft_train_cons,
                                y=microsoft_train_Y)

dfm_microsoft_test_cons<-TASSL_dfm(gd_microsoft_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_microsoft_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
microsoft_test_predict_msft<-predict(microsoft_model_cons,
                                     newx = dfm_microsoft_test_cons)[,1]
# estimate accuracy
cons_acc_msft<-kendall_acc(microsoft_test_predict_msft,microsoft_test_Y)


dfm_microsoft_test_cons<-TASSL_dfm(gd_microsoft_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
microsoft_test_predict_amzn<-predict(amazon_model_cons,
                                     newx = dfm_microsoft_test_cons)[,1]
# estimate accuracy
cons_acc_amzn<-kendall_acc(microsoft_test_predict_amzn,microsoft_test_Y)


cons_acc_msft

cons_acc_amzn


#################################################
# Question 9
#################################################

# 9. For this question, lets just focus on the six big tech companies. Use their company label as a
# the outcome of a six-category multinomial model. Predict the company for each review using
# ngrams extracted from the cons text boxes. Produce a confusion matrix and calculate the
# accuracy of your model. Write a few sentences describing the result.

gd_train_mn<-gd_train %>%
  filter(FAANG==1)

gd_test_mn<-gd_test %>%
  filter(FAANG==1)

gd_train_mn_dfm<-TASSL_dfm(gd_train_mn$cons,ngrams=1)
gd_test_mn_dfm<-TASSL_dfm(gd_test_mn$cons,ngrams=1,min.prop=0) %>%
  dfm_match(colnames(gd_train_mn_dfm))

# Multinomial tends to be a bit slower
gd_mn_model<-glmnet::cv.glmnet(x=gd_train_mn_dfm,
                               y=gd_train_mn$company,
                               family="multinomial",
                               maxit=5000)
plot(gd_mn_model)

# With type="class", you can get a single predicted label for each document
mn_predict_label<-predict(gd_mn_model,
                          newx = gd_test_mn_dfm,
                          type="class")[,1]

# raw accuracy
mean(mn_predict_label==gd_test_mn$company)

# Confusion matrix - great for multinomials!

table(mn_predict_label,gd_test_mn$company)

# to export the table more easily
table(mn_predict_label,gd_test_mn$company) %>%
  write.csv("mn_table.csv")


#################################################
# Question 10
#################################################

# Apply it to both text fields - pros and cons - and use the predictions to
# calculate the two accuracy scores. Write a few sentences about the results.

# The yelp star MODEL

trainX<-dfm_rev[train_split,]

trainY<-review_dat$stars[train_split]

testX<-dfm_rev[-train_split,]

testY<-review_dat$stars[-train_split]

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# accuracy
kendall_acc(testY,test_predict)


cons_dfm<-TASSL_dfm(gd_test$cons,ngrams=1,min.prop=0) %>%
  dfm_match(colnames(dfm_rev))

pros_dfm<-TASSL_dfm(gd_test$pros,ngrams=1,min.prop=0) %>%
  dfm_match(colnames(dfm_rev))

cons_predict<-predict(lasso_model,newx = cons_dfm)[,1]
pros_predict<-predict(lasso_model,newx = pros_dfm)[,1]

kendall_acc(gd_test$overall,cons_predict)
kendall_acc(gd_test$overall,pros_predict)


#################################################

# For the remaining questions, train a twelve-topic model in the cons of the training data.

drops<-(rowSums(dfm_amazon_train_cons)==0)

topic_train=dfm_amazon_train_cons[!drops,] %>%
  as.dfm() %>%
  convert(to="stm")

topic_docs=gd_amazon_train$cons[!drops]

topicMod<-stm(documents = topic_train$documents,
              vocab= topic_train$vocab,
              K=12,
              init.type = "Spectral")

saveRDS(topicMod,file="data/topicMod.RDS")

topicMod<-readRDS("data/topicMod.RDS")

topicNames<-paste0("Topic",1:topicNum)

#################################################
# Question 11
#################################################

# 11. Use findThoughts and labelTopics to learn what each topic is about. Come up with some
# labels to describe eight of the topics. Put those labels on a labelTopics plot, which shows the five
# most distinctive words (by FREX) for each topic.

topicNames[1]="Customer Service"
topicNames[4]="Managers"
topicNames[6]="Work Life Balance"
topicNames[7]="Hiring"
topicNames[9]="Benefits"
topicNames[10]="Overtime"
topicNames[11]="Word Speed"
topicNames[12]="Treatment"


# Most common topics, and most common words from each topic
plot(topicMod,type="summary",n = 7,xlim=c(0,.3),labeltype = "frex",
     topic.names = topicNames) 

#################################################
# Question 12
#################################################


cloud(topicMod,6)


findThoughts(model=topicMod,
             texts=topic_docs,
             topics=6,n = 2)
