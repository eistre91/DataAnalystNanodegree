library(ggplot2)
library(dplyr)

data(diamonds)

ggplot(aes(x=price), data=diamonds) +
  geom_histogram(aes(fill=cut)) +
  facet_wrap( ~ color) +
  scale_fill_brewer(type = 'qual') +
  scale_x_log10()

ggplot(aes(x=table, y=price), data=diamonds) +
  geom_point(aes(color=cut)) +
  scale_color_brewer(type = 'qual')


diamonds$volume <- with(diamonds, x*y*z) 
big_volume <- quantile(diamonds$volume, probs = 0.99)

ggplot(aes(x=x*y*z, y=price),
       data=subset(diamonds, volume < big_volume)) +
  geom_point(aes(color=clarity)) +
  scale_y_log10() +
  scale_color_brewer(type='div')

setwd('C:/DataAnalystNanodegree/Project4')
pf <- read.delim('pseudo_facebook.tsv')
pf$prop_initiated <- with(pf, ifelse(friend_count > 0, friendships_initiated/friend_count, NA))

pf$year_joined <- floor(2014 - pf$tenure / 365)
pf$year_joined.bucket <- cut(pf$year_joined, breaks=c(2004,2009,2011,2012,2014))

ggplot(aes(x=tenure, y=prop_initiated), data=pf) +
  geom_line(aes(color=year_joined.bucket), stat='summary', fun.y='median')

ggplot(aes(x=round(tenure/7)*7, y=prop_initiated), data=pf) +
  geom_line(aes(color=year_joined.bucket), stat='summary', fun.y='median')

pf %>%
  group_by(year_joined.bucket) %>%
  filter(!is.na(prop_initiated)) %>%
  summarise(mean_prop_initiated = mean(prop_initiated),
            n = n())

ggplot(aes(x = cut, y = price/carat), data = diamonds) +
  geom_jitter(aes(color=color)) +
  facet_wrap( ~ clarity) +
  scale_color_brewer(type='div')
