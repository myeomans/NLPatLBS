################################################
#
#                  ML & NLP @ LBS
#
#                Activity 1 Answers
#
#
################################################

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
source("kendall_acc.R")

###############################################################
###############################################################

# Here  I am creating a function that saves all of our defaults in one place
TAB_dfm<-function(text,
                  ngrams=1:2,
                  stop.words=TRUE,
                  min.prop=.01){
  if(!is.character(text)){                # First, we check our input is correct
    stop("Must input character vector")
  }
  drop_list=""
  if(stop.words) drop_list=stopwords("en") #uses stop.words argument to adjust what is dropped
  
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

###############################################################
# Load data
###############################################################

# Review data
review_data<-readRDS("data/reviews.RDS")

# Business data
businesses<-readRDS("data/businessset.RDS")
# First thing - check variables

names(review_data)

names(businesses)

businesses<-businesses %>%
  # remove the ones we don't need
  filter(business_id%in%review_data$business_id) %>%
  # One variable name overlaps, so we rename one
  rename(average_stars="stars") %>%
  # convert to numeric 
  mutate(price=as.numeric(RestaurantsPriceRange2))


# We want to use reviews to predict price data, but price is in businesses, not reviews

# To move the business data over to the review data, we use left_join

reviews <- review_data %>%
  left_join(businesses,
            by="business_id")

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(reviews),round(nrow(reviews)/2))


reviews_train<-reviews[train_split,]
reviews_test<-reviews[-train_split,]

###############################################################
# Q1 - THE GENDER MODEL
###############################################################

trainX<-TAB_dfm(reviews_train$text)

testX<-TAB_dfm(reviews_test$text,min.prop = 0) %>%
  dfm_match(colnames(trainX))

# Extract Y variables
trainY<-reviews %>%
  slice(train_split) %>%
  pull(male)

testY<-reviews %>%
  slice(-train_split) %>%
  pull(male)


# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# calculate accuracy

kendall_acc(test_predict,testY)

#####################
# Build a plot
#####################

plotDat<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
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
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Gender Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


###############################################################
# Q2 - TRANSFER LEARNING
###############################################################


# extract data from different contexts in both test and training data
reviews_train_AM<-reviews_train %>%
  filter(grepl("Breakfast",categories)&(!grepl("Nightlife",categories)))

reviews_train_PM<-reviews_train %>%
  filter(grepl("Nightlife",categories)&(!grepl("Breakfast",categories)))


reviews_test_AM<-reviews_test %>%
  filter(grepl("Breakfast",categories)&(!grepl("Nightlife",categories)))

reviews_test_PM<-reviews_test %>%
  filter(grepl("Nightlife",categories)&(!grepl("Breakfast",categories)))


# train model from the AM price training data
dfm_reviews_train_AM<-TAB_dfm(reviews_train_AM$text,ngrams=1:2) %>%
  convert(to="matrix")
AM_model<-cv.glmnet(x=dfm_reviews_train_AM,
                    y=reviews_train_AM$stars)

# generate predictions for Breakfast test data
dfm_reviews_test_AM<-TAB_dfm(reviews_test_AM$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_AM)) %>%
  convert(to="matrix")
model_AM_predict_AM<-predict(AM_model,
                             newx = dfm_reviews_test_AM)[,1]
AM_AM_acc<-kendall_acc(model_AM_predict_AM,reviews_test_AM$stars)


# generate predictions for Nightlife test data
dfm_reviews_test_PM<-TAB_dfm(reviews_test_PM$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_AM)) %>%
  convert(to="matrix")
model_AM_predict_PM<-predict(AM_model,
                             newx = dfm_reviews_test_PM)[,1]
AM_PM_acc<-kendall_acc(model_AM_predict_PM,reviews_test_PM$stars)

# train model from the Nightlife training data
dfm_reviews_train_PM<-TAB_dfm(reviews_train_PM$text,ngrams=1:2) %>%
  convert(to="matrix")
PM_model<-cv.glmnet(x=dfm_reviews_train_PM,y=reviews_train_PM$stars)

# generate predictions for Nightlife test data
dfm_reviews_test_PM<-TAB_dfm(reviews_test_PM$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_PM)) %>%
  convert(to="matrix")
model_PM_predict_PM<-predict(PM_model,
                             newx = dfm_reviews_test_PM)[,1]
PM_PM_acc<-kendall_acc(model_PM_predict_PM,reviews_test_PM$stars)


# generate predictions for AM price test data
dfm_reviews_test_AM<-TAB_dfm(reviews_test_AM$text,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_reviews_train_PM)) %>%
  convert(to="matrix")
model_PM_predict_AM<-predict(PM_model,
                             newx = dfm_reviews_test_AM)[,1]
PM_AM_acc<-kendall_acc(model_PM_predict_AM,reviews_test_AM$stars)



# Put the accuracy into a plot

bind_rows(AM_AM_acc,
          AM_PM_acc,
          PM_AM_acc,
          PM_PM_acc) %>%
  mutate(Training=c("Breakfast","Breakfast","Nightlife","Nightlife"),
         Test=c("Breakfast","Nightlife","Breakfast","Nightlife")) %>%
  ggplot(aes(x=Training,color=Test,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point(stat="identity",size=3,
             position=position_dodge(.2)) +
  geom_errorbar(width=.2,position=position_dodge(.2)) +
  theme_bw() +
  ylim(70,82) +
  labs(y="Accuracy",x="Training Data",y="Test Data") +
  theme(text=element_text(size=20),
        legend.background = element_rect(color="black"),
        legend.position = c(.85,.2))

# Model generalizes well, but there is some drop-off during transfer learning
# Accuracy is higher in general for nightlife-trained mode, due to larger N



plotDat_AM<-AM_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score)) %>%
  left_join(data.frame(ngram=colnames(dfm_reviews_train_AM),
                       freq=colMeans(dfm_reviews_train_AM))) %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat_AM %>%
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
  ggtitle("Coefficients in Breakfast Model") +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


plotDat_PM<-PM_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score)) %>%
  left_join(data.frame(ngram=colnames(dfm_reviews_train_PM),
                       freq=colMeans(dfm_reviews_train_PM))) %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat_PM %>%
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
  ggtitle("Coefficients in Nightlife Model") +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))



###############################################################
# Q3 - MULTINOMIAL
###############################################################

# Feature extraction is the same... n-grams

# Extract Y variables
trainY<-reviews %>%
  slice(train_split) %>%
  pull(price)

testY<-reviews %>%
  slice(-train_split) %>%
  pull(price)


# Multinomial tends to be a bit slower
price_cat_model<-glmnet::cv.glmnet(x=trainX,
                                   y=trainY,
                                   family="multinomial",
                                   maxit=5000)

plot(price_cat_model)

# With type="class", you can get a single predicted label for each document
cats_predict_label<-predict(price_cat_model,
                            newx = testX,
                            type="class")[,1]

# Confusion matrix - great for multinomials!
table(cats_predict_label,testY)

# raw accuracy
mean(cats_predict_label==testY)


price_lin_model<-glmnet::cv.glmnet(x=trainX,
                                   y=trainY)

lin_predict_label<-predict(price_lin_model,
                            newx = testX)[,1]

# have to round
lin_predict_label_cat<-round(lin_predict_label)

# some are predicting out of distribution! must fix
table(lin_predict_label_cat)

lin_predict_label_cat[lin_predict_label_cat==0]<-1
  
table(lin_predict_label_cat)  


# Confusion matrix
table(lin_predict_label_cat,testY)

# raw accuracy
mean(lin_predict_label_cat==testY)
