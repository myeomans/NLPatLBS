

############### Word Vectors

# The real word vector files are ~ 6GB - too big! This is a smaller version,
# containing only the 50,000 most common words
vecSmall<-readRDS("data/vecSmall.RDS")

# Word frequency file - to reweight common words
load("data/wfFile.RData")

# one column with words, and 300 with vector projections (uninterpretable!)
head(vecSmall[,1:20])

# Some of my own home-brewed functions for vector calculation
source("vectorFunctions.R")

# Calculating similarity using bag of words doesn't know the difference between sad and happy!
bowSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled")

# However, processing the text as dense vectors allows the meaning to emerge. 
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall)

# Train a vector classifier
vdat<-vecCheck(rev_small$text,vecSmall,wfFile)

train_split=sample(1:nrow(rev_small),round(nrow(rev_small)/2))

lasso_vec<-glmnet::cv.glmnet(x=vdat[train_split,],
                             y=rev_small$stars[train_split])

# looks good
plot(lasso_vec)

test_vec_predict<-predict(lasso_vec,newx = vdat[-train_split,])

# could be better
cor.test(rev_small$stars[-train_split],test_vec_predict)


# Clear big files out of the workspace to reduce memory load
rm(vdat,vecSmall,wfFile)


############### Politeness

# Politeness requires SpaCyR, which requires SpaCy

# run only once, on a new machine, to install
#spacyr::spacy_install()

# run every session to initialize
spacyr::spacy_initialize()

# Some sample data to see the functions in action
gtest<-c("I understand that's what you mean.",
         "I don't understand you.",
         "It's not bad. I feel the same way.",
         "It's bad. I feel the same way.",
         "I'm sorry. But I don't agree with you about the Boris Johnson plan.",
         "I'm not sorry. But I agree with the New York plan.")


# Notice the dependency relations, part of speech tags, and name entities extracted by SpaCyR
spacyr::spacy_parse(gtest,dependency=TRUE,entity = TRUE)

# Note the different politeness features picked up, the negation handling, etc... 
politeness::politeness(gtest,parser="spacy",drop_blank=TRUE)


# Calculate politeness counts in reviews - use the pre-saved file for now

# rev_polite<-politeness(rev_small$text,parser="spacy")
# saveRDS(rev_polite,file="data/rev_polite.RDS")

rev_polite<-readRDS("data/rev_polite.RDS")

obviousgender=(!is.na(rev_small$user_male))&(abs(rev_small$user_male-.5)>.4)

# Looks like a big difference in overall word count... 
politenessPlot(rev_polite %>%
                 filter(obviousgender),
               rev_small$user_male[obviousgender],
               middle_out=.05,
               drop_blank = 0,
               split_levels = c("Female","Male"))

# Confirmed... Men write shorter texts
rev_small %>%
  with(summary(lm(word_count~user_male)))

# A quick way to fix - divide every column by a user's word count! Scale to average word count
rev_polite_av=as.data.frame(apply(rev_polite,2,function(x) mean(rev_small$word_count)*x/rev_small$word_count))

# Re-plot with averages
politenessPlot(rev_polite_av %>%
                 filter(obviousgender),
               rev_small$user_male[obviousgender],
               middle_out=.05,
               split_levels = c("Female","Male")) +
  # Note that politenessPlot can be customized like any normal ggplot with the + sign
  scale_y_continuous(name = "Feature Count per Average-length text",
                     breaks= c(.1,.5,1,2,5,10),
                     trans = "sqrt")

ggsave("genderreview.png")

# Men also give lower reviews... interesting!
rev_small %>%
  with(summary(lm(stars~user_male)))

