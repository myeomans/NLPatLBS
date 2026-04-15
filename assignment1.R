################################################
#
#                  ML & NLP @ LBS
#
#                    Activity 1
#
#
################################################

# Run these once, if you haven't installed them before
#install.packages("tidyverse")
# install.packages("quanteda")
# install.packages("textclean")
# install.packages("ggrepel")
# install.packages("glmnet")

# Run these every time
library(tidyverse) # useful for almost everything
library(quanteda) # text analysis workhorse
library(textclean) # extra pre-processing
library(ggrepel) # for plots
library(glmnet) # Our estimation model
source("kendall_acc.R") # accuracy function
######### Simple bag of words

testDocs<-c("This is a test sentence.", 
            "I am providing another sentence to test this.",
            "This isn't a sentence",
            "This is a test document. It has 2 sentences")

# First we need to split up the sentences into "tokens" - (usually words)

testDocs %>%
  tokens()

# Note the pipe works by adding the object as the first argument on the next line

# This is identical to the above:
tokens(testDocs)


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

# Here  I am creating a function that saves all of our defaults in one place
TAB_dfm<-function(text,
                  ngrams=1:2,
                  stop.words=TRUE,
                  min.prop=.01){
  if(!is.character(text)){                # First, we check our input is correct
    stop("Must input character vector")
  }
  drop_list=""
  if(stop.words) drop_list=stopwords("en") #uses stop.words arugment to adjust what is dropped
  
  text_data<-text %>%
    replace_contraction() %>%
    tokens(remove_numbers=TRUE,
           remove_punct = TRUE) %>%
    tokens_wordstem() %>%
    tokens_select(pattern = drop_list, 
                  selection = "remove") %>%
    tokens_ngrams(ngrams) %>%
    dfm() %>%
    dfm_trim(min_docfreq = min.prop,docfreq_type="prop")
  return(text_data)
}

TAB_dfm(dox)

# we can easily modify the defaults of our custom arguments
TAB_dfm(dox, ngrams=2)

TAB_dfm(dox, stop.words = FALSE)

TAB_dfm(dox, min.prop=.25)

# Note... this is a bit rudimentary
# If you prefer, you can use a more robust function I wrote for a different package
# install.packages("doc2concrete")
library(doc2concrete)

ngramTokens(dox)

###############################################################
######### New data - restaurant reviews
###############################################################
# Review data
review_dat<-readRDS("data/reviews.RDS") 

# Business data
businesses<-readRDS("data/businessset.RDS")
# First thing - check variables

names(reviews)

names(businesses)

businesses<-businesses %>%
  # remove the ones we don't need
  filter(business_id%in%review_dat$business_id) %>%
  # One variable name overlaps, so we rename one
  rename(average_stars="stars") %>%
  # convert to numeric 
  mutate(price=as.numeric(RestaurantsPriceRange2))


# We want to use reviews to predict price data, but price is in businesses, not reviews

# To move the business data over to the review data, we use left_join

reviews <- review_dat %>%
  left_join(businesses,
            by="business_id")

names(reviews)

# Calculate a 1-gram feature count matrix for the review data, with no dropped words
dfm1<-TAB_dfm(reviews$text,
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

table(reviews$price)

# Let's only use 1-grams for now
dfm2<-TAB_dfm(reviews$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# Lots of words
dim(dfm2)

# What we really care about is - does the presence of a word predict price?

# A simple start - correlate each word with star rating

correlations<-dfm2 %>%
  summarise_all(~round(cor(.,reviews$price),3)) %>%
  unlist()

# Ten lowest associations
sort(correlations)[1:10]

# Ten highest associations
rev(sort(correlations))[1:10]


# As we said in class we are not often interested in the effects of individual words
# Instead, we care more about how all the words perform as a class

# To do this, we will use the cv.glmnet() function to build a model

# First, we need to split the data into training and testing samples
set.seed(02138) # this makes sure the random split is the same every time
train_split=sample(1:nrow(reviews),round(nrow(reviews)/2))

table(train_split)

# create our prediction variables

reviews_train<-reviews[train_split,]
reviews_test<-reviews[-train_split,]


trainX<-TAB_dfm(reviews_train$text)

testX<-TAB_dfm(reviews_test$text)

# Notice we get different numbers of columns!
# Different columns are filtered out by min.prop in training and testing
dim(testX)
dim(trainX)

# For test data, set min.prop = 0 to keep all features in
# then use dfm_match() to coerce the test data to match the train data
testX<-TAB_dfm(reviews_test$text,min.prop = 0) %>%
  dfm_match(colnames(trainX))

# now they have the same columns :)
dim(testX)
dim(trainX)

# Extract Y variables
trainY<-reviews %>%
  slice(train_split) %>%
  pull(price)

testY<-reviews %>%
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

# We could just split the predictions in two, using the median

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

# We can also do accuracy using probability of superiority, aka Kendall's Tau

kendall_acc(test_predict,testY)

# The interpretation is "percent of paiwise comparisons in the correct direction
# It's non-parametric, and more common in machine learning world
# since distributions often have weird shapes in unstructured data

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
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))



######################################################################
# A multinomial classifier example
######################################################################

# the categories are in a text field, so we need to extract them - with dfm!

train_cats<-TAB_dfm(reviews_train$categories)%>%
  convert(to="data.frame") %>%
  as_tibble() %>%
  select(chines,sandwich,nightlif,mexican) 

# 3437 that are in only one category ... let's dump the rest
table(rowSums(train_cats))

one_cat_train=reviews_train %>%
  filter(rowSums(train_cats)==1) %>%
  mutate(category=case_when(
    str_detect(categories,"Chinese") ~ "chinese",
    str_detect(categories,"Sandwich") ~ "sandwich",
    str_detect(categories,"Nightlife") ~ "nightlife",
    str_detect(categories,"Mexican") ~ "mexican"
  ))

table(one_cat_train$category)

# do the same in the test set
test_cats<-TAB_dfm(reviews_test$categories)%>%
  convert(to="data.frame") %>%
  as_tibble() %>%
  select(chines,sandwich,nightlif,mexican) 


one_cat_test=reviews_test %>%
  filter(rowSums(test_cats)==1)%>%
  mutate(category=case_when(
    str_detect(categories,"Chinese") ~ "chinese",
    str_detect(categories,"Sandwich") ~ "sandwich",
    str_detect(categories,"Nightlife") ~ "nightlife",
    str_detect(categories,"Mexican") ~ "mexican"
  ))

table(one_cat_test$category)


# Feature extraction is the same... n-grams

one_cat_dfm_train<-TAB_dfm(one_cat_train$text,ngrams=1)

one_cat_dfm_test<-TAB_dfm(one_cat_test$text,
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
head(cats_predict,20)
dim(cats_predict)


######################################################################
# Transfer Learning example
######################################################################

# extract data from different contexts in both test and training data
reviews_train_low<-reviews_train %>%
  filter(price==1)

reviews_train_high<-reviews_train %>%
  filter(price>2)


reviews_test_low<-reviews_test %>%
  filter(price==1)

reviews_test_high<-reviews_test %>%
  filter(price>2)


# train model from the low price training data
dfm_reviews_train_low<-TAB_dfm(reviews_train_low$text,ngrams=1:2) %>%
  convert(to="matrix")
low_model<-cv.glmnet(x=dfm_reviews_train_low,
                     y=reviews_train_low$stars)

# generate predictions for low price test data
# note we have to use dfm_match to make sure the dfm columns are identical
dfm_reviews_test_low<-TAB_dfm(reviews_test_low$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_low)) %>%
  convert(to="matrix")
model_low_predict_low<-predict(low_model,
                                      newx = dfm_reviews_test_low)[,1]
# estimate accuracy
low_low_acc<-kendall_acc(model_low_predict_low,reviews_test_low$stars)


# generate predictions for high price test data
# note we have to use dfm_match to make sure the dfm columns are identical
dfm_reviews_test_high<-TAB_dfm(reviews_test_high$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_low)) %>%
  convert(to="matrix")
model_low_predict_high<-predict(low_model,
                                newx = dfm_reviews_test_high)[,1]
# estimate accuracy
low_high_acc<-kendall_acc(model_low_predict_high,reviews_test_high$stars)

low_low_acc
low_high_acc


# train model from the high price training data
dfm_reviews_train_high<-TAB_dfm(reviews_train_high$text,ngrams=1:2) %>%
  convert(to="matrix")
high_model<-cv.glmnet(x=dfm_reviews_train_high,
                     y=reviews_train_high$stars)

# generate predictions for high price test data
# note we have to use dfm_match to make sure the dfm columns are identical
dfm_reviews_test_high<-TAB_dfm(reviews_test_high$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_high)) %>%
  convert(to="matrix")
model_high_predict_high<-predict(high_model,
                               newx = dfm_reviews_test_high)[,1]
# estimate accuracy
high_high_acc<-kendall_acc(model_high_predict_high,reviews_test_high$stars)


# generate predictions for low price test data
# note we have to use dfm_match to make sure the dfm columns are identical
dfm_reviews_test_low<-TAB_dfm(reviews_test_low$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_high)) %>%
  convert(to="matrix")
model_high_predict_low<-predict(high_model,
                                newx = dfm_reviews_test_low)[,1]
# estimate accuracy
high_low_acc<-kendall_acc(model_high_predict_low,reviews_test_low$stars)

high_high_acc
high_low_acc


# Put the accuracy into a plot

bind_rows(low_low_acc,
          low_high_acc,
          high_low_acc,
          high_high_acc) %>%
  mutate(Training=c("Low Price","Low Price","High Price","High Price"),
         Test=c("Low Price","High Price","Low Price","High Price")) %>%
  ggplot(aes(x=Training,color=Test,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point(stat="identity",size=3,
             position=position_dodge(.2)) +
  geom_errorbar(width=.2,position=position_dodge(.2)) +
  theme_bw() +
  ylim(70,80) +
  labs(y="Accuracy",x="Training Data",y="Test Data") +
  theme(text=element_text(size=20),
        legend.background = element_rect(color="black"),
        legend.position = c(.85,.2))

# A little bit of drop-off in both transfer learning directions
# Mostly the same model, but some nuances 


### Plot models, compare coefficients


plotDat_low<-low_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score)) %>%
  left_join(data.frame(ngram=colnames(dfm_reviews_train_low),
                       freq=colMeans(dfm_reviews_train_low))) %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat_low %>%
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
  ggtitle("Coefficients in Low-Price Model") +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))



plotDat_high<-high_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score)) %>%
  left_join(data.frame(ngram=colnames(dfm_reviews_train_high),
                       freq=colMeans(dfm_reviews_train_high))) %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat_high %>%
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
  ggtitle("Coefficients in High-Price Model") +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))
