library(readr)
library(jsonlite)


bus_lines <- read_lines("yelp_academic_dataset_business.json", n_max = 1000000, progress = FALSE)

bus_combined <- str_c("[", str_c(bus_lines, collapse = ", "), "]")

busdatall<-fromJSON(bus_combined) %>%
  filter(grepl("Restaurants",categories)) %>%
  select(-hours,-is_open,-address)

busdat<-cbind(busdatall,busdatall$attributes)
busdat$attributes<-NULL

busdat<-busdat[,(colMeans(is.na(busdat))<.5)] %>%
  filter(state%in%c("CO","GA","MA","OH","FL","TX"))


rm(bus_combined,bus_lines,busdatall)


for(z in 1:87){
  review_lines <- read_lines("yelpfull.json", skip=(z-1)*100000,
                             n_max = 100000, progress = FALSE)
  reviews_comb <- str_c("[", str_c(review_lines, collapse = ", "), "]")
  reviews_dat <- fromJSON(reviews_comb)
  reviews_dat <- reviews_dat %>%
    filter(reviews_dat$business_id%in%busdat$business_id)
  if(z==1){reviews=reviews_dat
  } else{
    reviews<-bind_rows(reviews,reviews_dat)
  }
  print(paste(c(z,dim(reviews_dat),dim(reviews)),collapse=" "))
}

bID<-reviews %>%
  group_by(business_id) %>%
  summarize(n=n()) %>%
  filter(n>24)

reviews=reviews %>%
  filter(business_id%in%bID$business_id)  %>%
  group_by(business_id) %>%
  mutate(count=sample(1:n())) %>%
  ungroup() %>%
  filter(count<26) %>%
  select(-count)
  

rm(review_lines,reviews_comb,reviews_dat,bID)


coverage=0
rowa=1

while(coverage<1){
  user_lines <- read_lines("users.json", skip=(rowa-1)*100000,
                             n_max = 100000, progress = FALSE)
  user_comb <- str_c("[", str_c(user_lines, collapse = ", "), "]")
  user_dat <- fromJSON(user_comb)
  user_dat <- user_dat %>%
    filter(user_dat$user_id%in%reviews$user_id)
  if(rowa==1){user=user_dat
  } else{
    user<-bind_rows(user,user_dat)
  }
  coverage=mean(reviews$user_id%in%user$user_id)
  print(paste(rowa,round(coverage,2),collapse=" "))

  rowa=rowa+1
}

reviews<- reviews %>%
  left_join(user %>%
  select(user_id,user_name="name",user_review_count="review_count",
         user_yelping_since="yelping_since",
         user_average_stars="average_stars"))



quickdf<-ngramTokens(reviews$text,ngrams=1)


reviews<- reviews %>%
  filter(rowSums(quickdf)>10 & rowSums(quickdf)<600)

reviews$word_count<-str_count(reviews$text,"[[:alpha:]]+")

ambiences=c("touristy","hipster","romantic","divey","intimate",
           "trendy","upscale","classy","casual")

for(a in ambiences){
  busses[,paste0("ambience_",a)]<-1*grepl(paste0("'",a,"': True"),busses$Ambience)
}


meals=c("dessert","latenight","lunch","dinner","brunch","breakfast")

for(m in meals){
  busses[,paste0("good_for_",m)]<-1*grepl(paste0("'",m,"': True"),busses$GoodForMeal)
}

hist(reviews$user_male)


reviews1<-reviews[1:292000,]
reviews2<-reviews[292001:nrow(reviews),]

saveRDS(reviews1,file="data/reviewset1.RDS")
saveRDS(reviews2,file="data/reviewset2.RDS")

saveRDS(busdat,file="data/businessset.RDS")

rm(quickdf,reviews)

