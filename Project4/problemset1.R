library(ggplot2)
data(diamonds)

ggplot(aes(x=price), data=diamonds) +
  geom_histogram(binwidth=50) +
  scale_x_continuous(limits=c(250,1500))

table(diamonds$price < 500)
table(diamonds$price < 250)
table(diamonds$price >= 15000)

ggplot(aes(x=price), data=diamonds)+
  geom_histogram() +
  facet_wrap(~cut)

by(diamonds$price, diamonds$cut, summary)
by(diamonds$price, diamonds$cut, max)
by(diamonds$price, diamonds$cut, min)

qplot(x = price, data = diamonds) + facet_wrap(~cut, scales='free_y')

qplot(x = price/carat, data=diamonds) +
  facet_wrap(~cut) +
  scale_x_log10()

ggplot(aes(x=cut, y=price), data=diamonds) +
  geom_boxplot()

by(diamonds$price, diamonds$cut, summary)

ggplot(aes(x=color, y=price/carat), data=diamonds) +
  geom_boxplot()

ggplot(aes(x = carat), data=diamonds) +
  geom_freqpoly(binwidth = .01) +
  coord_cartesian(ylim=c(0,2500))

subset(names(table(diamonds$carat)), table(diamonds$carat) > 2000)

