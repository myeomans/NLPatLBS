#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#                  Assignment 1                     #
#                                                   #
#                                                   #
#####################################################

library(tidyverse) # useful for almost everything
library(quanteda) # text analysis workhorse
library(textclean) # extra pre-processing
library(ggrepel) # for plots
library(glmnet) # Our estimation model
library(pROC)  # binary prediction accuracy
library(doc2concrete) # ngramTokens
library(sentimentr) # sentiment
library(stm) # topic models

source("kendall_acc.R")

# note - this makes the "random" splits identical for all of us, so we get the same results
set.seed(20138)

#################################################
#################################################
# Part 1 - DFM basics
#################################################
#################################################


######### Simple bag of words

testDocs<-c("This is a test sentence.", 
            "I am providing another sentence to test this.",
            "This isn't a sentence",
            "This is a test document. It has 2 sentences")


# a quick word on tidyverse - the %>% is called "pipe"
# it takes the finished object from the current line
# and inserts it as the first argument to the function on the next line

# so, these two commands are identical
testDocs %>%
  tokens()

tokens(testDocs)


# Anyways, first we need to split up the sentences into "tokens" - (usually words)

testDocs %>%
  tokens()

# We then count how often each token occurs in each document 
# This produces a "document feature matrix" (or document term matrix)
# One row for each doc, one column for each feature
testDocs %>%
  tokens() %>%
  dfm()

# We can also combine adjoining words into "bigrams"

testDocs %>%
  tokens() %>%
  tokens_ngrams(2) %>%
  dfm()

# often people combine multiple token lengths together, as ngrams
testDocs %>%
  tokens() %>%
  tokens_ngrams(1:2) %>%
  dfm()

# Many different ways to tokenize - see the help file for options

?tokens

# We can stem words

testDocs %>%
  tokens(remove_punct=TRUE) %>%
  tokens_wordstem()

# we can remove punctuation
testDocs %>%
  tokens(remove_punct=TRUE) %>%
  tokens_ngrams(1:2)

# we can remove numbers
testDocs %>%
  tokens(remove_numbers=TRUE) %>%
  tokens_ngrams(1:2)

# contractions are done with a function from textclean
testDocs %>%
  replace_contraction() %>%
  tokens()


# dfm converts everything to lower case by default, but we can turn this off
testDocs %>%
  tokens() %>%
  dfm()

testDocs %>%
  tokens() %>%
  dfm(tolower=FALSE)

# we can also remove "stop words"
testDocs %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  tokens_ngrams(1:2)

# This is the built-in quanteda stopword list
stopwords("en")

# we can create our own custom list if we like
testDocs %>%
  tokens() %>%
  tokens_select(pattern = c("a","is","the"), 
                selection = "remove") %>%
  tokens_ngrams(1:2)


# Instead of removing common words, we can downweight them, using tfidf

dox<-c("This is a sentence.",
       "this is also a sentence.",
       "here is a rare word",
       "here is another word.",
       "and other sentences")

# Without tfidf, all words are given the same weight
dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Here, rare words are given more weight
dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_tfidf() %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# We can also remove words that are too rare to learn anything about

dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = 2) %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Usually we do this by proportion of words

dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = .25,docfreq_type="prop") %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Typically the cut-off gets set around 1% of documents

# Here  I am loading a function that saves all of our defaults in one place
source("TASSL_dfm.R")


TASSL_dfm(dox)

# we can easily modify the defaults of our custom arguments
TASSL_dfm(dox, ngrams=2)

TASSL_dfm(dox, stop.words = FALSE)

TASSL_dfm(dox, min.prop=.25)

# Note... this is a bit rudimentary
# If you prefer, you can use a more robust function I wrote for a different package
# install.packages("doc2concrete")
library(doc2concrete)

ngramTokens(dox)

#################################################
#################################################
# Part 2 - Yelp data
#################################################
#################################################


######### New data - restaurant reviews

# Review data
review_dat<-readRDS("data/review_dat.RDS") %>%
  filter(str_count(text,"[[:alpha:]]+")>25)

names(review_dat)


# Calculate a 1-gram feature count matrix for the review data, with no dropped words
dfm1<-TASSL_dfm(review_dat$text,
                ngrams=1,
                min.prop=0,
                stop.words = FALSE)

dim(dfm1) # >10k ngrams! Too many

# most common words - obvious
sort(colMeans(dfm1),decreasing=TRUE)[1:20]

# least common words
sort(colMeans(dfm1))[1:20]

######## Ok, let's build a model to predict price!

# First, let's look at our price data

table(review_dat$price)

# Let's only use 1-grams for now
dfm3<-TASSL_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# Lots of words
dim(dfm3)

#  Most common words in 1- and 2-price reviews... lots of the same words!
sort(colMeans(dfm3[review_dat$price==2,]),decreasing=T)[1:20]

sort(colMeans(dfm3[review_dat$price==1,]),decreasing=T)[1:20]

# What we really care about is - does the presence of a word predict price?

# A simple start - correlate each word with star rating

correlations<-dfm3 %>%
  summarise_all(~round(cor(.,review_dat$price),3)) %>%
  unlist()

# Ten lowest associations
sort(correlations)[1:10]

# Ten highest associations
rev(sort(correlations))[1:10]

# note - same as:
sort(correlations,decreasing=TRUE)[1:10]

# As we said in class we are not often interested in the effects of individual words
# Instead, we care more about how all the words perform as a class

# To do this, we will use the cv.glmnet() function to build a model

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(review_dat),round(nrow(review_dat)/2))

length(train_split)

# create our prediction variables
dfm3<-TASSL_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)


trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(price)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(price)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# let's plot the cross-validation curve to see if it's finding any signal
plot(lasso_model)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# Note that while the true answers are binary, the predictions are continuous
# Always check these distributions!!
hist(testY)
hist(test_predict)

# For now, let's just split the predictions in two, using the median

test_predict_binary=ifelse(test_predict>median(test_predict),
                           2,
                           1)
hist(test_predict_binary)

# quick plot of the split to make sure it looks right
plot(x=test_predict,y=test_predict_binary)


# This should have the same values as testY
hist(test_predict_binary)

# and we can calculate accuracy from that

round(100*mean(test_predict_binary==testY),3)

#### What is in the model? We can extract the coefficients

# lots of zeros
lasso_model %>%
  coef() %>%
  drop()

# let's get this in a data frame
lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".")

# just the top
lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  head(20)

# drop zeros, and save
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

plotCoefs

# create a similar data frame with ngram frequencies
plotFreqs<-data.frame(ngram=colnames(trainX),
                      freq=colMeans(trainX))


# combine data, round for easy reading
plotDat<-plotCoefs %>%
  left_join(plotFreqs) %>%
  mutate_at(vars(score,freq),~round(.,3))

head(plotDat)

# here's our first plot, with minimal customization
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  geom_point()

# Problems:
# Bad axis labels
# no point labels
# I don't like the default grey background
# legend is redundant

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  geom_point() +
  geom_label() +
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none")

# More problems:
# wasted space in Y axis
# lots of overlapping labels
# small axis labels
# i don't like the default colors

# colors we can set manually

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="blue",
                        mid = "grey",
                        high="green",
                        midpoint = 0)+
  geom_point() +
  geom_label_repel()+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

# let's get more words on the plot
# also make the X axis clearer
# use darker colors

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 50)+  
  scale_x_continuous(limits = c(-.2,.1),
                     breaks = seq(-.2,.2,.05)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

#################################################
#################################################
# Part 3 - glassdoor data
#################################################
#################################################

# new data! let's explore
glassdoor<-readRDS("data/glassdoorReviews.RDS")


# Only a few companies in this data
glassdoor %>%
  with(table(company))

# Split in categories - some big companies, some smaller ones
glassdoor %>%
  with(table(company,FAANG))

# Other important metadata - Overall rating
glassdoor %>%
  with(hist(overall))

# More exploring.... do companies differ by overall rating?
overall_avgs<-glassdoor %>%
  group_by(company) %>%
  summarize(m=mean(overall),
            se=sd(overall)/sqrt(n())) 

# note how we calculate a standard error above
# it is included through ymin and ymax on line 55

overall_avgs %>%
  ggplot(aes(x=company,color=company,
             y=m,ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar(width=.2) +
  theme_bw() +
  coord_flip() + # coord_flip makes the axis labels readable!
  scale_y_continuous(limits = c(3,5)) +
  labs(y="Overall Rating")+
  theme(legend.position="none")

# Let's explore the text.... 

# Note - there are two different text boxes!! "pros" and "cons"
glassdoor <- glassdoor%>%
  mutate(pros_wordcount=str_count(pros,"[[:alpha:]]+"),
         cons_wordcount=str_count(cons,"[[:alpha:]]+"))

# for showing a single continuous variable, we use a histogram
glassdoor %>%
  ggplot(aes(x=pros_wordcount)) +
  geom_histogram(bins = 100) +
  theme_bw() +
  xlim(0,100)

glassdoor %>%
  ggplot(aes(x=cons_wordcount)) +
  geom_histogram(bins = 100) +
  theme_bw() +
  xlim(0,100)

# Let's focus on people who actually wrote text in both boxes

gd_small<-glassdoor %>%
  filter(pros_wordcount>5 & cons_wordcount>5)

dim(gd_small)
# Even that's too big so let's get it down to 40,000 texts

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

# check the tuning to see if there is useful information
plot(amazon_model_pros)

##################################################################

# let's apply our model to two test sets

# We need the same X features in the test as in training 

# we use dfm_match() to make sure they are the same features

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

# check distributions - continuous predictor, continuous outcome
hist(amazon_test_predict_pros)
hist(amazon_test_Y)

# estimate accuracy - use kendall's tau
pros_acc<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

pros_acc

############################################

# Let's apply the same model to the cons text

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,ngrams=1:2,
                                min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_cons)[,1]

hist(amazon_test_predict_cons)
hist(amazon_test_Y)

# estimate accuracy
cons_acc<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

# why is accuracy so low?
cons_acc

# Combine accuracy estimates for a plot
bind_rows(pros_acc %>%
            mutate(field="Pros ngrams"),
          cons_acc %>%
            mutate(field="Cons ngrams")) %>%
  ggplot(aes(x=field,color=field,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        panel.grid=element_blank(),
        legend.position="none")

#################################################

# Back to the Yelp data 

train_split=sample(1:nrow(review_dat),9000)

review_dat_train<-review_dat[train_split,]
review_dat_test<-review_dat[-train_split,]


review_dat_dfm_train<-TASSL_dfm(review_dat_train$text,ngrams=1)

review_dat_dfm_test<-TASSL_dfm(review_dat_test$text,
                            ngrams=1,
                            min.prop=0) %>%
  dfm_match(colnames(review_dat_dfm_train))


rev_model<-glmnet::cv.glmnet(x=review_dat_dfm_train %>%
                               as.matrix(),
                             y=review_dat_train$stars)

plot(rev_model)

#### Evaluate Accuracy
test_ngram_predict<-predict(rev_model,
                            newx = review_dat_dfm_test %>%
                              as.matrix())[,1]

acc_ngram<-kendall_acc(review_dat_test$stars,test_ngram_predict)

acc_ngram

############ Find examples

# store predictions in data, calculate accuracy
review_dat_test<-review_dat_test %>%
  mutate(prediction=test_ngram_predict,
         error=abs(stars-prediction),
         bias=stars-prediction)

close_high<-review_dat_test %>%
  filter(stars==5 & error<.5) %>%
  select(text,stars,prediction)

close_low<-review_dat_test %>%
  filter(stars==1 & error<.5) %>%
  select(text,stars,prediction)

close_high
close_high %>%
  slice(1:2) %>%
  pull(text)

close_low
close_low %>%
  slice(1:2) %>%
  pull(text)

# Error analysis - find biggest misses

review_dat_test %>%
  ggplot(aes(x=prediction)) +
  geom_histogram()

review_dat_test %>%
  ggplot(aes(x=stars)) +
  geom_histogram()

miss_high<-review_dat_test %>%
  arrange(bias) %>%
  slice(1:10) %>%
  select(text,stars,prediction)

miss_low<-review_dat_test %>%
  arrange(-bias) %>%
  filter(stars==5) %>%
  slice(1:10) %>%
  select(text,stars,prediction)

miss_low
miss_low%>%
  slice(1:2) %>%
  pull(text)

miss_high
miss_high%>%
  slice(3) %>%
  pull(text)


############### Benchmarks

# Create benchmarks

review_dat_test <- review_dat_test %>%
  mutate(text_wdct=str_count(text,"[[:alpha:]]+"),
         model_random=sample(test_ngram_predict),
         sentiment=sentiment_by(text)$ave_sentiment)

acc_wdct<-kendall_acc(review_dat_test$stars,
                      -review_dat_test$text_wdct)

acc_wdct



acc_random<-kendall_acc(review_dat_test$stars,
                        review_dat_test$model_random)

acc_random


acc_sentiment<-kendall_acc(review_dat_test$stars,
                           review_dat_test$sentiment)

acc_sentiment

######################################################################
# A multinomial classifier example
######################################################################

# the categories are in a text field, so we need to extract them - with dfm!

train_cats<-TASSL_dfm(review_dat_train$categories)%>%
  convert("data.frame") %>%
  select(chines,sandwich,nightlif,mexican) 

# 4432 that are in only one category ... let's dump the rest
table(rowSums(train_cats))

one_cat_train=review_dat_train %>%
  filter(rowSums(train_cats)==1) %>%
  mutate(category=case_when(
    str_detect(categories,"Chinese") ~ "chinese",
    str_detect(categories,"Sandwich") ~ "sandwich",
    str_detect(categories,"Nightlife") ~ "nightlife",
    str_detect(categories,"Mexican") ~ "mexican"
  ))

table(one_cat_train$category)

# do the same in the test set
test_cats<-TASSL_dfm(review_dat_test$categories)%>%
  convert("data.frame") %>%
  select(chines,sandwich,nightlif,mexican) 


one_cat_test=review_dat_test %>%
  filter(rowSums(test_cats)==1)%>%
  mutate(category=case_when(
    str_detect(categories,"Chinese") ~ "chinese",
    str_detect(categories,"Sandwich") ~ "sandwich",
    str_detect(categories,"Nightlife") ~ "nightlife",
    str_detect(categories,"Mexican") ~ "mexican"
  ))

table(one_cat_test$category)


# Feature extraction is the same... n-grams

one_cat_dfm_train<-TASSL_dfm(one_cat_train$text,ngrams=1)

one_cat_dfm_test<-TASSL_dfm(one_cat_test$text,
                            ngrams=1,
                            min.prop=0) %>%
  dfm_match(colnames(one_cat_dfm_train))



# Multinomial tends to be a bit slower
one_cat_model<-glmnet::cv.glmnet(x=one_cat_dfm_train,
                                 y=one_cat_train$category,
                                 family="multinomial",
                                 maxit=5000)

plot(one_cat_model)

# With type="class", you can get a single predicted label for each document
cats_predict_label<-predict(one_cat_model,
                            newx = one_cat_dfm_test,
                            type="class")[,1]

# raw accuracy
mean(cats_predict_label==one_cat_test$category)

# Confusion matrix - great for multinomials!

table(cats_predict_label,one_cat_test$category)

# to export the table more easily
table(cats_predict_label,one_cat_test$category) %>%
  write.csv("cats_table.csv")

# type="response" produces a probability that each document is in each class
cats_predict<-predict(one_cat_model,
                      newx = one_cat_dfm_test,
                      type="response")[,,1] %>%
  round(4)

# this way you can set different thresholds for each label
# use the probabilities in a regression instead of the absolute labels, etc.

# returns a matrix - one row per document, one column per class
head(cats_predict)
dim(cats_predict)


######################################################################
# A topic model example
######################################################################

# First we need a dfm object (ngram matrix in a quanteda file format)
# Topic models are usually estimated with only unigrams, and without stopwords

# There is one row that is empty! The topic model with break with this
table(rowSums(review_dat_dfm_train)==0)

# You should remove it first, before estimating 
review_dat_train<-review_dat_train[rowSums(review_dat_dfm_train)!=0,]
review_dat_dfm_train<-review_dat_dfm_train[rowSums(review_dat_dfm_train)!=0,]

table(rowSums(review_dat_dfm_test)==0)

# You should remove it first, before estimating 
review_dat_test<-review_dat_test[rowSums(review_dat_dfm_test)!=0,]
review_dat_dfm_test<-review_dat_dfm_test[rowSums(review_dat_dfm_test)!=0,]


# Train a 20-topic model
rev_topicMod20<-stm(review_dat_dfm_train,K=20,init.type = "Spectral")

# There are metrics you can use to choose the topic number.
# These are controversial... you are better off adjusting to taste
# This is how you would run that, though....
# Fist convert to stm format, then put the documents and vocab into searchK()

# rev_stm_format<-review_dat_dfm_train %>%
#   convert(to="stm")
# sk<-searchK(rev_stm_format$documents,
#             rev_stm_format$vocab,
#             K=c(10,20,30,40))
# plot(sk)

# Note - you can save topic models as RDS files, too!

saveRDS(rev_topicMod20,file="data/rev_topicMod20.RDS")


rev_topicMod20<-readRDS("data/rev_topicMod20.RDS")

topicNum=rev_topicMod20$settings$dim$K

# LDA will not name your topics for you! It's good to come up with some names on your own

topicNames<-paste0("Topic",1:topicNum)

# Most common topics, and most common words from each topic
plot(rev_topicMod20,type="summary",n = 7,xlim=c(0,.3),labeltype = "frex",
     topic.names = topicNames) 

# You can add names to the vector one at a time
topicNames[1]="Tourist"
topicNames[2]="Breakfast"
topicNames[4]="Value"
topicNames[6]="Dessert"
topicNames[12]="Service"
topicNames[13]="Seafood"
topicNames[17]="Sushi"
topicNames[18]="Booking"
topicNames[20]="Cafe"
# We can also grab more words per topic
labelTopics(rev_topicMod20)

findThoughts(model=rev_topicMod20,
             texts=review_dat_train$text,
             topics=1,n = 5)

# We can even put them in a word cloud! If you fancy it

cloud(rev_topicMod20,19)

cloud(rev_topicMod20,13)

# Which topics correlate with one another?
plot(topicCorr(rev_topicMod20),
     vlabels=topicNames,
     vertex.size=20)

stmEffects<-estimateEffect(1:topicNum~stars,
                           rev_topicMod20,
                           meta= review_dat_train %>%
                             select(stars))


# The default plotting function is bad... Here's another version
bind_rows(lapply(summary(stmEffects)$tables,function(x) x[2,1:2])) %>%
  mutate(topic=factor(topicNames,ordered=T,
                      levels=topicNames),
         se_u=Estimate+`Std. Error`,
         se_l=Estimate-`Std. Error`) %>%
  ggplot(aes(x=topic,y=Estimate,ymin=se_l,ymax=se_u)) +
  geom_point() +
  geom_errorbar() +
  coord_flip() +
  geom_hline(yintercept = 0)+
  theme_bw() +
  labs(y="Correlation with Star Rating",x="Topic") +
  theme(panel.grid=element_blank(),
        axis.text=element_text(size=20))

# Update to their package means this isn't working... 
#
# but you shouldn't use it anyways 
# 
# # This contains the topic proportions for each document..
# topic_prop_train<-rev_topicMod20$theta
# dim(topic_prop_train)
# colnames(topic_prop_train)<-topicNames
# 
# # We can use these topic proportions just like any other feature
# rev_model_stm<-glmnet::cv.glmnet(x=topic_prop_train,
#                                 y=review_dat_train$stars)
# 
# # Note that we didn't give enough features... there is no U shape
# plot(rev_model_stm)
# 
# review_dat_dfm_test
# 
# 
# review_dat_dfm_tp<-textProcessor(review_dat_test$text)
# 
# review_dat_dfm_pd<-prepDocuments(review_dat_dfm_tp$documents,
#                               review_dat_dfm_tp$vocab)
# 
# review_dat_dfm_ac<-alignCorpus(review_dat_dfm_pd,
#                             old.vocab=colnames(review_dat_dfm_train))
#                      
# 
# # 
# topic_prop_test<-fitNewDocuments(rev_topicMod20,
#                                  review_dat_dfm_test %>%
#                                    convert(to="stm") %>%
#                                    `$`(documents))
# 
# 
# test_stm_predict<-predict(rev_model_stm,
#                           newx = topic_prop_test$theta)[,1]
# 
# # Note the drop in performance, compared to the ngrams
# acc_stm<-kendall_acc(review_dat_test$stars,test_stm_predict)
# 
# acc_stm
# 
