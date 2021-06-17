#######################################
# Natural Language for Social Science #
#        Prof Michael Yeomans         #
#######################################

# Run only once, when you first download the data
install.packages("tidyverse","quanteda","politeness",
                 "doc2concrete","glmnet","ggrepel","stm",
                 "syuzhet","sentimentr","doMC","spacyr")

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


########################################################
# Data wrangling.... ignore for now
########################################################
# source("dataload.R") # to trim yelp data from raw JSON files... don't run, it's slow! (but if you're curious.... )
# source("datashrink.R") # This shrinks the big dataset to something more manageable.

########################################################

# Let's load the small data from memory
rev_small<-readRDS("data/rev_small.RDS")
bus_small<-readRDS("data/bus_small.RDS")

# Class 1!   # ngrams, model training, dictionaries

source("basicNLP.R")    



# source("structuralNLP.R") #  vectors, politeness, accommodation
# 
# source("receptiveness.R") # receptiveness example
