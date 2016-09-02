library(ggplot2)
data(diamonds)

ggplot(aes(x = x, y = price), data=diamonds) +
  geom_point()

?diamonds

with(diamonds, cor.test(price, x))
with(diamonds, cor.test(price, y))
with(diamonds, cor.test(price, z))

ggplot(aes(x = depth, y = price), data=diamonds) +
  geom_point(alpha = 1/20) +
  scale_x_continuous(limits=c(55,70))

ggplot(data = diamonds, aes(x = depth, y = price)) + 
  geom_point(alpha=1/100) +
  scale_x_continuous(breaks=seq(0, 80, 2),
                     limits=c(54,72))

with(diamonds, cor.test(depth, price))


ggplot(aes(x=carat, y=price), data=diamonds) +
  geom_point() +
  xlim(0, quantile(diamonds$carat, probs=.99)) +
  ylim(0, quantile(diamonds$price, probs=.99))


diamonds$volume <- diamonds$x * diamonds$y * diamonds$z
with(diamonds, volume <- x * y * z)

ggplot(aes(x=volume, y=price), data=diamonds) +
  geom_point()

with(diamonds, cor.test(price, volume))
with(subset(diamonds, volume > 0 & volume <= 800), cor.test(price, volume))

ggplot(aes(x=volume, y=price), data=subset(diamonds, volume > 0 & volume <= 800)) +
  geom_point(alpha=1/100) +
  geom_smooth (method="lm", color="red") +
  ylim(0,20000)

library(dplyr)

diamondsByClarity <- diamonds %>%
  group_by(clarity) %>%
  summarise(mean_price = mean(price),
            median_price = median(price),
            min_price = min(price),
            max_price = max(price),
            n = n()) %>%
  arrange(clarity)

diamonds_by_clarity <- group_by(diamonds, clarity)
diamonds_mp_by_clarity <- summarise(diamonds_by_clarity, mean_price = mean(price))

diamonds_by_color <- group_by(diamonds, color)
diamonds_mp_by_color <- summarise(diamonds_by_color, mean_price = mean(price))

library(gridExtra)

bp1 <- ggplot(aes(x=color, y=mean_price), data=diamonds_mp_by_color) +
  geom_bar(stat="identity")

bp2 <- ggplot(aes(x=clarity, y=mean_price), data=diamonds_mp_by_clarity) +
  geom_bar(stat="identity")
  
grid.arrange(bp1, bp2, ncol=2)
