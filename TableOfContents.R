#######################################
# Natural Language for Social Science #
#        Prof Michael Yeomans         #
#######################################

# Run only once, when you first download the data
install.packages(c("tidyverse","quanteda","politeness",
                 "doc2concrete","glmnet","ggrepel","stm",
                 "syuzhet","sentimentr","doMC","spacyr"))

library(tidyverse) # Contains MANY useful functions
library(textclean) # contraction handler
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

# Functions for use in the class
source("kendall_acc.R")    
source("vectorFunctions.R")    
source("TASSL_dfm.R")

# These are introductions to basic tidyverse - plotting and string handling
# If you are new to tidyverse, please take a few minutes to go through these!!
source("text_basics.R")    
source("ggplot_tips.R")   

# Class 1!  ngrams, model training, categories
source("NLP_LBS1.R")    
source("NLP_LBS1_answers.R")    

# Class 2!  dictionaries, embeddings, sentence structure
source("NLP_LBS2.R")    
source("NLP_LBS2_answers.R")    

