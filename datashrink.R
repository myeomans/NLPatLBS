
# These are the full datasets - don't load them for now! But they will come in handy later
busses<-readRDS("data/businessset.RDS") # 35,086 restaurants from Yelp
reviews<-bind_rows(readRDS("data/reviewset1.RDS"), # 584,137 restaurant reviews
                   readRDS("data/reviewset2.RDS")) # it's big! had to split in two for github

# There are way too many examples for us... Let's trim it down
# This is the only city I've lived in from the data (no London or Toronto, alas!)
# We'll also only look at restaurants that are 2/4 on the price range (second cheapest!)
# And we'll drop reviews where a gender cannot be identified
bus_small<-busses %>%
  filter(city=="Cambridge" & RestaurantsPriceRange2==2)
rev_small <- rev_small %>%
  filter(business_id%in%bus_small$business_id & !is.na(user_male))

# Save data as we go
saveRDS(rev_small,file="data/rev_small.RDS")
saveRDS(bus_small,file="data/bus_small.RDS")
