

#######################################
# Natural Language for Social Science #
#######################################

library(tidyverse) # Contains MANY useful functions
library(quanteda)  # many useful text analysis tools
library(politeness) # structural features from text
library(doc2concrete) # Contains the ngramTokens function - mostly a wrapper around quanteda tools
library(glmnet) # A simple machine learning model (the LASSO)
library(ggrepel) # useful for plotting later on
library(stm)  # High-performance topic modelling
library(syuzhet) # a benchmark for sentiment detection
library(sentimentr) # a benchmark for sentiment detection
library(doMC) # to speed up some code with parallel processing
library(spacyr) # to parse grammar



#Get Data
rev_small<-readRDS("data/rev_small.RDS")
bus_small<-readRDS("data/bus_small.RDS")

rev_small$word_count<-str_count(rev_small$text,"[[:alpha:]]+")

# Distribution of word counts
rev_small %>%
  ggplot(aes(x=word_count)) +
  geom_histogram()


dfm1<-ngramTokens(rev_small$text,ngrams=1)

dim(dfm1)

## COMMON WORDS - MOST AND LEAST
sort(colMeans(dfm1),decreasing=TRUE)[1:20]
sort(colMeans(dfm1))[1:20]


## RARE WORDS
dfm1All<-ngramTokens(rev_small$text[1:1000],ngrams=1,sparse = 1)
sort(colSums(dfm1All)[colSums(dfm1All)==1])
rm(dfm1All)



### PREDICT GENDER


# Let's use 1-, 2- and 3-grams
dfm3<-ngramTokens(rev_small$text,ngrams=1:3)

dim(dfm3)

## SPLIT THE SAMPLE
train_split=sample(1:nrow(rev_small),round(nrow(rev_small)/2))

## LASSO TO PREDICT
lasso_mod<-glmnet::cv.glmnet(x=dfm3[train_split,],
                             y=rev_small$user_male[train_split])

### SAVE MODELS TO MEMORY
saveRDS(lasso_mod,file="data/modLASSO.RDS")
saveRDS(train_split,file="data/train_split.RDS")


plot(lasso_mod)




## TEST THE MODEL

## Predict from test sample
test_predict<-predict(lasso_mod,newx = dfm3[-train_split,])


## Get true gender
test_actual<-rev_small$user_male[-train_split]

## Pearson correlation
cor.test(test_predict,test_actual)


## WITH OTHER BENCHMARKS

sentiment_one<-syuzhet::get_sentiment(rev_small$text[-train_split],method="nrc")

cor.test(sentiment_one,test_actual)

sentiment_two<-syuzhet::get_sentiment(rev_small$text[-train_split],method="bing")

cor.test(sentiment_two,test_actual)


sentiment_three<-sentimentr::sentiment(rev_small$text[-train_split]) %>%
  group_by(element_id) %>%
  summarize(sent=mean(sentiment))

cor.test(sentiment_three$sent,test_actual)





############ Interpreting the model with lists and plots

# Extract coefficients of model into a table
scoreSet<-coef(lasso_mod) %>%
  as.matrix() %>%
  data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score="X1") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score)) %>%
  left_join(data.frame(ngram=colnames(dfm3),
                       freq=colMeans(dfm3)))

# 10 words that predict low scores
scoreSet%>%
  arrange(score) %>%
  slice(1:10)

# 10 ngrams that predict low scores
scoreSet%>%
  arrange(-score) %>%
  slice(1:10)

#combine coefficients with ngram frequencies

scoreSet %>%
  # can't plot everything... this mutate line removes some labels
  # that are not common (>1%) or distinctive enough 
  mutate(ngram=ifelse((abs(score)>.01)&(freq>.005),ngram,"")) %>%
  # let's add a bit of color
  mutate(col=case_when(
    score>.005  ~ "blue",
    score<(-.005) ~ "red",
    T ~ "black")) %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=col)) +
  scale_color_manual(breaks=c("blue","black","red"),
                     values=c("blue","black","red"))+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 30,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  scale_x_continuous(breaks=seq(-.4,.4,.1),
                     labels = seq(-.4,.4,.1),
                     limits = c(-.25,.25))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


# Save the plot for a slide deck
ggsave("wordz.png",units="cm",dpi=200,width=35,height=20)

rev_small %>%
  bind_cols(dfm3 %>%
              as_tibble() %>%
              select(husband,wife,boyfriend,girlfriend)) %>%
  mutate(Gender=ifelse(user_male>.5,"male","female")) %>%
  filter(abs(user_male-.5)>.3)%>%
  group_by(Gender) %>%
  summarize(husband=mean(husband),
            wife=mean(wife),
            boyfriend=mean(boyfriend),
            girlfriend=mean(girlfriend))  %>%
  pivot_longer(-Gender,names_to="Relation",values_to="Use") %>%
  mutate(ci=1.96*sqrt(Use*(1-Use)/nrow(rev_small))) %>%
  ggplot(aes(x=Relation,y=Use,ymin=Use-ci,ymax=Use+ci,
             group=Gender,fill=Gender)) +
    geom_bar(stat="identity",position="dodge")+
    geom_errorbar(position=position_dodge(1),width=.3)+
    coord_flip()+
    theme_bw()+
  labs(x="Use rate per review") +
  theme(legend.position=c(.7,.3),
        legend.text=element_text(size=18),
        axis.text=element_text(size=18))
### COMMON TOPICS
#st we use stm to estimate the topic model


# First we need a dfm object (ngram matrix in a quanteda file format)
# Topic models are usually estimated with only unigrams, and without stopwords
dfmTPX<-as.dfm(ngramTokens(rev_small$text,ngrams=1,stop.words = FALSE))
# #
topicMod20<-stm(dfmTPX,K=20)
#
topicMod30<-stm(dfmTPX,K=30)

# Note - you can save topic models as RDS files, too!
saveRDS(topicMod20,file="data/topicMod20.RDS")
saveRDS(topicMod30,file="data/topicMod30.RDS")

# Two models - 20 and 30 topics (K is *very* hard to choose)

topicMod20<-readRDS("data/topicMod20.RDS")
#topicMod30<-readRDS("data/topicMod30.RDS")

######## Let's focus on the 20 topic model for now...

# Most common topics, and mst common words from each topic
plot(topicMod20,type="summary",n = 7,xlim = c(0,.3)) 

# We can also grab more words per topic
labelTopics(topicMod20)

# Estimate effects of topics and star rating
ee<-estimateEffect(1:20~user_male,topicMod20,
                   meta= rev_small[,c("stars","user_male")])

# The default plotting function is bad... Here's another version
bind_rows(lapply(summary(ee)$tables,function(x) x[2,1:2])) %>%
  mutate(topic=factor(paste("Topic",1:20),ordered=T,
                      levels=paste("Topic",1:20)),
         se_u=Estimate+`Std. Error`,
         se_l=Estimate-`Std. Error`) %>%
  ggplot(aes(x=topic,y=Estimate,ymin=se_l,ymax=se_u)) +
  geom_point() + 
  geom_errorbar() +
  coord_flip() +
  geom_hline(yintercept = 0)+
  theme_bw()

# Which topics correlate with one another?
plot(topicCorr(topicMod20))

# This contains the topic proportions for each document..
topic_prop<-topicMod20$theta
dim(topic_prop)


lasso_stm<-glmnet::cv.glmnet(x=topic_prop[train_split,],
                             y=rev_small$user_male[train_split])

# Note that we didn't give enough features... there is no U shape
plot(lasso_stm)

test_stm_predict<-predict(lasso_stm,newx = topic_prop[-train_split,])

# Note the small drop in performance, compared to the ngrams
cor.test(rev_small$user_male[-train_split],test_stm_predict)


