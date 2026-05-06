################################################
#
#                  ML & NLP @ LBS
#
#                    Activity 3
#
#
################################################

library(quanteda)
library(ggrepel)
library(textclean)
library(tidyverse)
library(glmnet)
library(sentimentr) # we didn't get to this last week... 
library(spacyr) # a new one!
library(politeness) # another new one! And a package I wrote :)
library(semgram) # also new - for motifs

# some extra machine learning libraries
library(randomForest)
library(xgboost)
library(rpart)
library(rpart.plot)
library(ranger)
library(SHAPforxgboost)

source("TAB_dfm.R")
source("kendall_acc.R")
source("modelPlot.R") # for coefficient plots



reviews<-readRDS(file="data/reviews.RDS")

train_split=sample(1:nrow(reviews),9000)

reviews_train<-reviews[train_split,]
reviews_test<-reviews[-train_split,]


######################################################################
# Let's get sentiment working
######################################################################

text_sample=reviews_test %>%
  slice(1:100) %>%
  pull(text)

# the base function does each sentence separately
sentiment(text_sample)

#sentiment_by() calculates once per document
# but then you need the $ave_sentiment column
sentiment_by(text_sample)

#this is the sentiment vector you want
text_sample %>%
  sentiment_by() %>%
  pull(ave_sentiment)

# Note that when the vector gets large, it will throw an annoying warning
# technically, they want you to add an extra step

text_sample %>%
  get_sentences() %>%
  sentiment_by() %>%
  pull(ave_sentiment)

# BUT it will do that step automatically even if you don't do it yourself

#############################################
# More sentiment code
#############################################

# Accuracy score using traditional dictionary

NRC<-dictionary(list(pos=lexicon::hash_sentiment_nrc %>% 
                  filter(y==1) %>% 
                  pull(x),
                neg=lexicon::hash_sentiment_nrc %>% 
                  filter(y==-1) %>% 
                  pull(x)))

reviews_train_dicts<-reviews_train %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(NRC) %>%
  convert(to="data.frame")

reviews_train_dicts <- reviews_train_dicts %>% 
  mutate(NRC_sentiment=pos-neg) 

kendall_acc(reviews_train_dicts$NRC_sentiment,
            reviews_train$stars)


# Lots of dictionaries!
lexicon::available_data()

########################################################
# using L&M dictionary - with/without grammar awareness
########################################################

loughran_words<-textdata::lexicon_loughran()

reviews_train_dicts<-reviews_train %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words)) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# all the dictionaries are in there!
head(reviews_train_dicts)

# usually you want to divide by the word count
reviews_train_dicts<-reviews_train_dicts %>%
  mutate_all(~./reviews_train$word_count)

reviews_train_dicts<-reviews_train_dicts %>%
  mutate(sentiment=positive-negative)

kendall_acc(reviews_train_dicts$sentiment,
            reviews_train$stars)

reviews_train<-reviews_train %>%
  mutate(LMsentiment=sentiment_by(text,
                                  polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) %>%
           pull(ave_sentiment))

kendall_acc(reviews_train$LMsentiment,
            reviews_train$stars)

# domain-general dictionary works better than finance
reviews_train<-reviews_train %>%
  mutate(BLsentiment=sentiment_by(text) %>%
           pull(ave_sentiment))

kendall_acc(reviews_train$BLsentiment,
            reviews_train$stars)

# examples - 
c("this is a bad product","this is not a bad product") %>%
  sentiment_by(polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) 


c("this is a bad product","this is not a bad product") %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words))

# vader also works this way
library(vader)

vader::vader_df(text = c("this is a bad product","this is not a bad product"))



# vader is good for quirks of internet speech
vader::vader_df(text = c("lol this is not bad",
                         ":) this is not bad",
                         "this is not bad"))

reviews_train<-reviews_train %>%
  slice(1:1000) %>%
  mutate(VDsentiment=vader_df(text) %>%
           pull(compound))

kendall_acc(reviews_train$VDsentiment,
            reviews_train$stars)

################################################
################################################
#     an introduction to some spacy features
################################################


###################################
set.seed(02138)

reviews_train<-reviews[1:1000,]
reviews_test<-reviews[2000:3000,]


spacyr::spacy_initialize()


# Politeness

reviews_train_polite<-politeness(reviews_train$text,parser="spacy")

featurePlot(reviews_train_polite,
            reviews_train$stars,
            split_levels = c("Low","High"),
            split_name = "Stars",
            middle_out = .05) +
  theme(panel.grid = element_blank()) 


p_model<-cv.glmnet(x=as.matrix(reviews_train_polite),
                   y=reviews_train$stars)

reviews_test_polite<-politeness(reviews_test$text,parser="spacy")

test_predict<-predict(p_model,
                      newx = as.matrix(reviews_test_polite))[,1]

acc_polite<-kendall_acc(reviews_test$stars,test_predict)

acc_polite


rev_tiny <- reviews %>%
  slice(1:500)


rev_tiny_sp<-spacy_parse(rev_tiny$text,
                         nounphrase = T,
                         lemma = T,
                         dependency = T,
                         pos = T,
                         tag=T)

head(rev_tiny_sp,20)

# Save the output of slow-executing code!!
saveRDS(rev_tiny_sp,"data/rev_tiny_sp.RDS")

rev_tiny_sp<-readRDS("data/rev_tiny_sp.RDS")

##################################################
# Use lemmas instead of stems!
##################################################

# recreate documents from the lemmas
rev_lemma_docs<-rev_tiny_sp %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

#extract lemmas as words from the document
lemmas<-rev_lemma_docs$text %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  dfm() %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

# the normal approach of stemming
stems<-TAB_dfm(rev_lemma_docs$text) %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

#lots of shortened non-words
stems[!stems%in%lemmas][1:100]

#this makes sense at least
lemmas[!lemmas%in%stems][1:100]

##################################################
# Using POS tags to disambiguate words
##################################################

# words with two senses
two_senses<-rev_tiny_sp %>%
  group_by(token,pos) %>%
  summarize(pos_ct=n()) %>%
  left_join(rev_tiny_sp %>%
              group_by(token) %>%
              summarize(all_ct=n())) %>%
  mutate(pos_ratio=pos_ct/all_ct) %>%
  filter(all_ct>5) %>%
  filter(pos_ratio>.2 & pos_ratio<.8) %>%
  as.data.frame()

# a few examples of words with multiple POS
two_senses

rev_sp_tagged <- rev_tiny_sp %>%
  left_join(two_senses %>%
              mutate(token_tag=paste0(token,"_",pos)) %>%
              select(token,pos,token_tag)) %>%
  mutate(tagged_tokens=ifelse(is.na(token_tag),token,token_tag))

# create a dfm from this
rev_tagged_docs<-rev_sp_tagged %>%
  group_by(doc_id) %>%
  summarize(text=paste(tagged_tokens, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

TAB_dfm(rev_tagged_docs$text) %>%
  colnames() %>%
  sort()


##################################################
# named entity recognition
##################################################

rev_ner<-spacy_extract_entity(rev_tiny$text)

rev_ner %>%
  filter(ent_type=="GPE") %>%
  with(rev(sort(table(text))))

rev_ner <- rev_ner %>%
  uncount(length) %>%
  group_by(doc_id,start_id) %>%
  mutate(doc_token_id=start_id+0:(n()-1),
         first=1*(start_id==doc_token_id)) %>%
  ungroup() %>%
  mutate(text=str_replace_all(text," ","_")) %>%
  select(doc_id,ner_text="text",first,doc_token_id) 

rev_sp_ner <- rev_tiny_sp %>%
  group_by(doc_id) %>%
  # annoying that the nounphrase counts doc tokens, not sentence tokens
  # but we do what we must
  mutate(doc_token_id=1:n()) %>%
  ungroup()%>%
  left_join(rev_ner) %>%
  filter(is.na(ner_text)|first==1) %>%
  mutate(ner_token=ifelse(is.na(ner_text),token,ner_text)) %>%
  select(-pos,-tag,-head_token_id,-first,-dep_rel,-nounphrase,-ner_text)

# generate a dfm from this

rev_ner_docs<-rev_sp_ner %>%
  group_by(doc_id) %>%
  summarize(text=paste(ner_token, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

# extract all the common noun phrases
phrases<-TAB_dfm(rev_ner_docs$text,
                 min.prop = .001) %>%
  convert(to="data.frame") %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

phrases[1:100]

rm(rev_sp_ner,rev_lemma_docs,
   rev_sp_tagged)

##################################################
# Extracting motifs 
##################################################

extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("server","waiter","waitress"),
               add_sentence = T,
               markup = T)



extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("husband","wife","spouse","partner"),
               add_sentence = T,
               markup = T)


extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("Toronto"),
               add_sentence = T,
               markup = T)


partners<-extract_motifs(tokens = rev_tiny_sp,
                         parse_multi_token_entities = T,
                         entities = c("husband","wife"),
                         add_sentence = T,
                         markup = T)


head(partners$actions)

partners$actions %>%
  group_by(action,Entity) %>%
  summarize(n=n()) %>%
  pivot_wider(names_from="Entity",values_from="n") %>%
  mutate_all(~replace_na(.,0)) %>%
  mutate(total=husband+wife,
         tilt=husband/(husband+wife)) %>%
  arrange(-tilt)



################################################
#
# Machine Learning Bonus
#
################################################


rests<-readRDS("data/businessset.RDS") %>%
  mutate(price=as.numeric(RestaurantsPriceRange2)) %>%
  filter(!is.na(price) & price<3)

cats=rests %>%
  pull(categories) %>%
  tolower() %>%
  str_replace_all("-","_") %>%
  str_replace_all(" & ","_") %>%
  str_replace_all(", ",",") %>%
  str_replace_all(" ,",",") %>%
  str_replace_all(" ","_") %>%
  str_replace_all(","," ") %>%
  tokens(remove_punct = T) %>%
  dfm() %>%
  dfm_trim(min_docfreq = .05,docfreq_type="prop") %>%
  as.matrix()

cleaner<-function(text){
  text=ifelse(is.na(text),"NA",text)
  text=gsub("u'","",text,fixed=T)
  text=gsub("'","",text,fixed=T)
  text=paste0("_",text)
  return(text)
}

catvars<-c("NoiseLevel","RestaurantsAttire","RestaurantsTakeOut",
           "HasTV","OutdoorSeating","Caters","RestaurantsReservations",
           "RestaurantsDelivery","GoodForKids")
pred_dat<-rests %>%
  select(price) %>%
  cbind(cats) %>%
  cbind(model.matrix(~.-1, # -1 removes the intercept
                     data=rests %>%
                       select(catvars) %>%
                       mutate_all(cleaner)))


set.seed(02138)
train_split=sample(1:nrow(pred_dat),20000)

rests_train<-pred_dat[train_split,]
rests_test<-pred_dat[-train_split,]


rests_train_x<-rests_train %>%
  select(-price) %>%
  as.matrix() %>%
  apply(1:2,as.numeric)

rests_test_x<-rests_test %>%
  select(-price) %>%
  as.matrix() %>%
  apply(1:2,as.numeric)


##########################################
# LASSO Benchmark
##########################################


lasso_mod<-cv.glmnet(x=rests_train_x,y=rests_train$price)

plot(lasso_mod)

lasso_pred_test<-predict(lasso_mod,newx = rests_test_x)[,1]

kendall_acc(rests_test$price,
            lasso_pred_test)

modelPlot(lasso_mod,rests_train_x) +
  labs(y='Feature count average')


# Ablation test - does "american" matter?

lasso_mod<-cv.glmnet(x=rests_train_x[,-4],y=rests_train$price)

plot(lasso_mod)

lasso_pred_test<-predict(lasso_mod,newx = rests_test_x[,-4])[,1]

kendall_acc(rests_test$price,
            lasso_pred_test)



# Ablation test - does "fast food" matter?

lasso_mod<-cv.glmnet(x=rests_train_x[,-17],y=rests_train$price)

plot(lasso_mod)

lasso_pred_test<-predict(lasso_mod,newx = rests_test_x[,-17])[,1]

kendall_acc(rests_test$price,
            lasso_pred_test)




# Ablation test - do reservations matter?
rests_train_nocats_x<-rests_train_x[,-(43:45)]
rests_test_nocats_x<-rests_test_x[,-(43:45)]


lasso_nocats_mod<-cv.glmnet(x=rests_train_nocats_x,y=rests_train$price)

plot(lasso_nocats_mod)

lasso_pred_nocats_test<-predict(lasso_nocats_mod,newx = rests_test_nocats_x)[,1]

kendall_acc(rests_test$price,
            lasso_pred_nocats_test)

##########################################
##########################################
# classification tree
##########################################
##########################################
treemod<-rpart(price~.,rests_train)

plot(treemod, margin = 0.2)
text(treemod, use.n = TRUE, cex = 0.8)

rpart.plot(treemod)

tree_pred_test<-predict(treemod,newdata=rests_test)

kendall_acc(rests_test$price,
            tree_pred_test)

###################################################
###################################################
# Random Forests
###################################################
# 
# rf<-randomForest(price~.,rests_train,
#                  sampsize=10000, # observations to test
#                  mtry=5, # number of considered variables at each node
#                  ntree=100) # number of trees in forest


rf<-ranger(price~.,rests_train,
           importance="impurity",
           num.trees=500) # number of trees in forest

importance(rf) %>%
  as.data.frame() %>%
  rownames_to_column("variable") %>%
  rename(importance=".") %>%
  mutate(variable=fct_reorder(variable,importance)) %>%
  ggplot(aes(x=variable,y=importance)) +
  geom_point() +
  theme_bw() +
  coord_flip()



rf_pred_test<-predict(rf,data=rests_test)

kendall_acc(rests_test$price,
            rf_pred_test$predictions)

####################################
# xgboost
####################################

xgbMod <- xgboost(data = rests_train_x, 
                  label = rests_train$price, 
                  
                  
                  # max.depth = 4, 
                  # eta = .3, 
                  # nthread = 10, 
                  nrounds = 1000, 
                  verbose=0)

xgb_pred_test<-predict(xgbMod, rests_test_x)

kendall_acc(rests_test$price,
            xgb_pred_test)

# Setting for boosting a linear model
# xgbMod <- xgboost(data = rests_train_x[,1:20], 
#                   label = rests_train$price, 
#                   booster="gblinear",
#                   nrounds = 1000, 
#                   verbose=0)

#######################################################
# SHAP for xgboost
#######################################################
# 
# Doesn't work any more, not sure why.....
# 
# xgbMod <- xgboost(data = rests_train_x[,1:10], 
#                   label = rests_train$price, 
#                   # max.depth = 4, 
#                   # eta = .3, 
#                   # nthread = 10, 
#                   nrounds = 1000, 
#                   verbose=0)
# 
# # get the values
# shap_values <- shap.values(xgb_model = xgbMod,
#                            X_train = rests_train_x[,1:10])
# shap_values$mean_shap_score
# 
# # beeswarm plots
# shap.plot.summary.wrap1(xgbMod, X = rests_train_x[,1:20])
# 
# shap_long <- shap.prep(xgb_model = xgbMod, 
#                        X_train = rests_train_x[,1:20])
# 
# # deeper plot for single feature - more useful for continuous variables
# shap.plot.dependence(data_long = shap_long, x = "nightlife") 

#######################################################
# Post-double-LASSO
#######################################################

# raw regression - lots of confounds
pred_dat %>%
  with(summary(lm(price~nightlife)))

pred_X<-pred_dat %>%
  select(-nightlife,-price) %>%
  as.matrix()

pred_D<-pred_dat$nightlife

pred_Y<-pred_dat$price


# 1. Selection of predictors for Y
y.lasso <- cv.glmnet(x = pred_X, y = pred_Y)
coef.y.lasso <- coef(y.lasso, s = "lambda.1se")
coef.y.label <- rownames(coef.y.lasso)[as.vector(!(coef.y.lasso == 0))]
# 2. Selection of predictors for D
d.lasso <- cv.glmnet(x = pred_X, y = pred_D)
coef.d.lasso <- coef(d.lasso, s = "lambda.1se")
coef.d.label <- rownames(coef.d.lasso)[as.vector(!(coef.d.lasso == 0))]

# 3. Refit the model
coef.double.label <- union(coef.y.label, coef.d.label)
coef.double.label<-coef.double.label[coef.double.label!="(Intercept)"]

dat.double <- data.frame(Y = pred_Y, D = pred_D) %>%
  cbind(pred_X[,coef.double.label])
post.double.lasso <- lm("Y ~ .", data = dat.double)

summary(post.double.lasso)



