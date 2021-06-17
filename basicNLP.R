
######### Simple bag of words

testDox<-c("this is a test sentence.", 
           "I am providing another sentence to test.",
           "this is a test document. It has two sentences")

# Default settings - 1-grams, stemmed words
ngramTokens(testDox,ngrams=1)

# 1-grams, no stemming (notice sentences and sentence in separate columns now)
ngramTokens(testDox,ngrams=1,wstem="none")

# removing stop words this time (there's a lot of them!)
ngramTokens(testDox,ngrams=1,stop.words=FALSE)

# now let's grab 3-grams - three-word sequences.
ngramTokens(testDox,ngrams=3)

# Finally, let's grab n-grams - all 1-, 2-, and 3-word sequences.
ngramTokens(testDox,ngrams=1:3)

######### New data - restaurant reviews

# The weird value in the second argument is called a "regular expression"
# Here it's just counting consecutive sets of letters i.e. words
rev_small$word_count<-str_count(rev_small$text,"[[:alpha:]]+")

# Distribution of word counts
rev_small %>%
  ggplot(aes(x=word_count)) +
  geom_histogram()


# Calculate a 1-gram feature count matrix for the review data
dfm1<-ngramTokens(rev_small$text,ngrams=1)

dim(dfm1)

# most common words - obvious
sort(colMeans(dfm1),decreasing=TRUE)[1:20]

# least common words (note - ngramTokens filters words that occur in less than 0.5% of documents)
sort(colMeans(dfm1))[1:20]

# If you turn off filtering of rare words...
dfm1All<-ngramTokens(rev_small$text[1:1000],ngrams=1,sparse = 1)

# many words only occur once! Not very useful....
sort(colSums(dfm1All)[colSums(dfm1All)==1])

# don't do this. let's get rid of rare words
rm(dfm1All)

######## Ok, let's build a model to predict sentiment!

# Let's use 1-, 2- and 3-grams
dfm3<-ngramTokens(rev_small$text,ngrams=1:3)

dim(dfm3)


# Here we build a model with our feature set

# First, randomly split into training and testing data
train_split=sample(1:nrow(rev_small),round(nrow(rev_small)/2))
# Put training data into LASSO model
lasso_mod<-glmnet::cv.glmnet(x=dfm3[train_split,],
                             y=rev_small$stars[train_split])

# You can save models, just like data!
# Highly recommended if the dataset is large and/or code is slow.
saveRDS(lasso_mod,file="data/modLASSO.RDS")
saveRDS(train_split,file="data/train_split.RDS")

# Then you can load from memory later
lasso_mod=readRDS("data/modLASSO.RDS")
train_split=readRDS("data/train_split.RDS")


# Did our model find anything? yes! big dip in out-of sample error
# Also note counts along top of graph - our model uses hundreds of features
plot(lasso_mod)


################ Evaluating accuracy of the model

# fit the trained model to the test data
test_predict<-predict(lasso_mod,newx = dfm3[-train_split,])

# Distribution of predictions seems reasonable (some out-of-range extremes, though)
hist(test_predict)

#grab the true star ratings from 
test_actual<-rev_small$stars[-train_split]


# Accuracy using pearson correlation... not bad!
cor.test(test_predict,test_actual)

# compared to what? Let's benchmark against some others

# basic word list annotated out of context by mTurk workers
sentiment_one<-syuzhet::get_sentiment(rev_small$text[-train_split],method="nrc")

# okay... not great
cor.test(sentiment_one,test_actual)

# word list from an ML model trained on movie reviews
sentiment_two<-syuzhet::get_sentiment(rev_small$text[-train_split],method="bing")

# better!
cor.test(sentiment_two,test_actual)


# this package uses sentence boundaries and handles negations... a bit smarter
sentiment_three<-sentimentr::sentiment(rev_small$text[-train_split]) %>%
  group_by(element_id) %>%
  summarize(sent=mean(sentiment))

# better still! But not as good as our in-domain model
cor.test(sentiment_three$sent,test_actual)


############ Interpreting the model with lists and plots

# Extract coefficients of model into a table
scoreSet<-coef(lasso_mod) %>%
  as.matrix() %>%
  data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score="X1") %>%
  # removes ngrams that did not get a score in the model
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score)) %>%
  # this left_join line adds ngram frequencies to the table
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
  mutate(ngram=ifelse((abs(score)>.05)&(freq>.01),ngram,"")) %>%
  # let's add a bit of color
  mutate(col=case_when(
    score>.02  ~ "blue",
    score<(-.02) ~ "red",
    T ~ "black")) %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=col)) +
  scale_color_manual(breaks=c("blue","black","red"),
                     values=c("blue","black","red"))+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 30,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  scale_x_continuous(breaks=seq(-.4,.4,.2),
                     labels = seq(-.4,.4,.2),
                     limits = c(-.5,.5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

# Save the plot for a slide deck
ggsave("wordz.png",units="cm",dpi=200,width=35,height=20)


############### Topic Models

# First we use stm to estimate the topic model


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
                             y=rev_small$stars[train_split])

# Note that we didn't give enough features... there is no U shape
plot(lasso_stm)

test_stm_predict<-predict(lasso_stm,newx = topic_prop[-train_split,])

# Note the small drop in performance, compared to the ngrams
cor.test(test_actual,test_stm_predict)

