


neg_turns<-read.csv("data/neg_turns.csv") 

neg_people<-read.csv("data/neg_people.csv")

# breaking up conversation into 60-second blocks
neg_turns$spanminute<-floor(neg_turns$span/60)

# distribution - mostly early, as expected
hist(neg_turns$spanminute)

# collapsing spread out text at end
neg_turns$spanminute[neg_turns$spanminute>15]<-15

neg_turns$wordcount=str_count(neg_turns$text,"[[:alpha:]]+")

# Calculate politeness, add it to the turn-level data

# neg_polite<-politeness(neg_turns$text,parser="spacy") 
# saveRDS(neg_polite,file="data/neg_polite.RDS")

# If SpaCy/politeness isn't working, you can load from saved file here
neg_polite<-readRDS("data/neg_polite.RDS")

neg_turns_all<-bind_cols(neg_turns,neg_polite)



### Add counts to person-level data

featureSums<-neg_turns_all %>%
  # first four turns of conversation
  filter(turn<5) %>%
  select(id,group,study,Hedges:Conjunction.Start) %>%
  group_by(id,study,group) %>%
  # take an count of everything
  summarize_all(list(sum))

head(featureSums)  

neg_people_plus <- neg_people %>%
  # merge feature counts into person level data
  left_join(featureSums) %>%
  mutate_at(names(neg_polite), replace_na, replace=0)

# Save a common filter - buyers-only - we can re-use in later code
plotFilter<-(neg_people_plus$seller==0)

politeness::politenessPlot(neg_people_plus %>%
                             filter(plotFilter) %>%
                             select(Hedges:Conjunction.Start),
                           neg_people_plus %>%
                             filter(plotFilter) %>%
                             select(bonus) %>%
                             unlist(),
                           middle_out=.1,
                           split_level=c("low bonus","high bonus"),
                           drop_blank=0)

# Now compare by randomized condition
politeness::politenessPlot(neg_people_plus %>%
                             filter(plotFilter) %>%
                             select(Hedges:Conjunction.Start),
                           neg_people_plus %>%
                             filter(plotFilter) %>%
                             select(tough) %>%
                             unlist(),
                           middle_out=.1,
                           drop_blank=0)

# treatment affects buyers
neg_people_plus %>%
  filter(seller==0) %>%
  with(summary(lm(bonus~study+tough)))


library(tidyverse)
library(politeness)

# This is data from an mTurk study we ran.. people wrote hypothetical first offers
head(phone_offers,20)

# The features that differ by condition in that data
politenessPlot(politeness(phone_offers$message,parser="spacy"),
               phone_offers$condition,
               split_levels = c("tough","warm"),
               middle_out = .1)

# We used it to train a classifier in an earlier paper
neg_people_plus$Warmth<-politenessProjection(df_polite_train = politeness(phone_offers$message,parser="spacy"),
                                             covar = phone_offers$condition,
                                             df_polite_test = neg_people_plus %>%
                                               select(Hedges:Conjunction.Start))$test_proj

# That predicts outcomes in this new data
neg_people_plus %>%
  filter(seller==0) %>%
  with(summary(lm(bonus~Warmth)))

# This also is manipulated across conditions
neg_people_plus %>%
  filter(seller==0) %>%
  with(summary(lm(Warmth~study+tough)))

# Here's a plot over time. 

# we use expand grid to create a full panel model - one cell for every minute-length span
plotData<-expand.grid(id=unique(neg_turns_all$id),
            spanminute=unique(neg_turns_all$spanminute)) %>%
  left_join(neg_turns_all %>%
              group_by(id) %>%
              summarize(seller=first(seller))) %>%
  left_join(neg_turns_all %>%
              group_by(id,spanminute) %>%
              summarize(wordcount=sum(wordcount))) %>%
  mutate(wordcount=replace_na(wordcount,0))
  
# Calculate group averages, standard errors, and get them into a plot
plotData %>%
  mutate(seller=ifelse(seller==1,"Seller","Buyer")) %>%
  group_by(spanminute,seller) %>%
  summarize(wordcount=mean(wordcount),
            se=sd(wordcount)/sqrt(n())) %>%
  ggplot(aes(y=wordcount,ymin=wordcount-se,ymax=wordcount+se,
             x=spanminute,group=seller,color=seller)) +
  geom_point() +
  geom_errorbar()+
  geom_line()+
  theme_bw()


# timeFraction is normalized by conversation length..  0 (beginning) to 1 (end)
plotData<-expand.grid(id=unique(neg_turns_all$id),
                      turnFrac=c(0:3)) %>%
  left_join(neg_turns_all %>%
              group_by(id) %>%
              summarize(seller=first(seller))) %>%
  left_join(neg_turns_all %>%
              #divide the timespans into quarters
              mutate(turnFrac=floor(timeFraction*4)) %>%
              group_by(id,turnFrac) %>%
              #choosing wordcount as our feature
              summarize(fcount=sum(wordcount))) %>%
  mutate(fcount=replace_na(fcount,0))


# Calculate group averages, standard errors, and get them into a plot
plotData %>%
  mutate(seller=ifelse(seller==1,"Seller","Buyer")) %>%
  group_by(turnFrac,seller) %>%
  summarize(avg=mean(fcount),
            se=sd(fcount)/sqrt(n())) %>%
  ggplot(aes(y=avg,ymin=avg-se,ymax=avg+se,
             x=turnFrac,group=seller,color=seller)) +
  geom_point(position=position_dodge(.2)) +
  geom_errorbar(position=position_dodge(.2),width=.3)+
  theme_bw()
