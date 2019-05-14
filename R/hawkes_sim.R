# a  = simulateHawkes(0.17454545, 0.05818182, 0.07757576, 40)
# print(length(a[[1]]))
# write.csv(a[[1]], file = "../practice/hsim.csv", row.names=FALSE)
# likelihoodHawkes(0.5, 0.9, 2, a[[1]])


# n_rep = 100
# t = 1000
# mu = 1.5
# hawkes_n = rep(0, n_rep)
# main_alpha = 120
# main_beta = 160
# 
# jumps = seq(1, 1000, by=5)
# print(length(jumps))
# 
# means = rep(0, length(jumps))
# expected = rep(0, length(jumps))
# 
# cnt = 1
# for (div in jumps){
#   alpha = main_alpha / div
#   beta = main_beta / div
#   
#   for (i in 1:n_rep){
#     a  = simulateHawkes(mu, alpha, beta, t)
#     hawkes_n[i] = length(a[[1]])
#   }
#   
#   num_expected = (mu * t) / (1 - alpha / beta)
#   num_actual = mean(hawkes_n)
#   
#   means[cnt] = num_actual
#   expected[cnt] = num_expected
#   
#   cnt = cnt + 1
#   print(cnt)
#   # cat("Mu", mu,  ", Alpha:",alpha, ", Beta:", beta, ", Ratio:", alpha/beta, "\n")
#   # cat("Expected # events:", num_expected, ", Actual:", num_actual, ", Percent diff:", 100 * (num_expected - num_actual) / num_expected, "\n")
# }
# 
# subtitle = paste("Precent Difference in Actual and Expected Number of Events \n", "Mu:", mu,  ", Starting alpha:", main_alpha, ", Starting Beta:", main_beta, ", Fixed Ratio:", alpha/beta, ", T:", t, "\n")
# p.diff = 100 * (expected - means) / expected
# plot(jumps, p.diff, xlab = "Alpha and Beta Divisor", ylab = "Percent Difference", main = subtitle)


library(ggplot2)
library(hawkes)
require(gridExtra)

plot_event_hist <- function(event_data, name, t){
  hist(event_data, main = paste(name, "Model Empirical Distribution of Number of Events \n T=", t), xlab="Number of Events", prob = TRUE)
  
  points(seq(min(event_data), max(event_data), length.out=5000),
         dnorm(seq(min(event_data), max(event_data), length.out=5000),
               mean(event_data), sd(event_data)), type="l", col="red")
  lines(density(event_data, adjust = 2), col = "blue")
  # Add a legend
  legend("topright", legend=c("Normal(mean, var)", "Empirical Density"),
         col=c("red", "blue"), lty=1)
}

plot_event_hist_log <- function(event_data, name, mu, alpha, beta, t){
  emp_mean = mean(event_data)
  emp_vari = var(event_data)
  
  asy_mean = (mu * t) / (1 - alpha / beta)
  asy_var = (mu * t) / (1 - alpha / beta) ** 3
  
  df = data.frame(event_data)
  plt <- ggplot(data=df, aes(x=df$event_data)) + 
  geom_histogram(col="black",
                 aes(y = ..density..),
                 alpha = .2,
                 binwidth=density(df$event_data)$bw) +
  geom_density(fill="red", alpha = 0.2) +
  stat_function(fun = dnorm, args = list(mean = asy_mean, sd = sqrt(asy_var)), color = "blue") + 
  scale_y_continuous(trans='log') +
  # labs(title=paste(name, "Model Empirical Distribution of Number of Events \nT=", t, 
  #                  "\nAsy Mean:", round(asy_mean, 3), " Var:", round(asy_var, 3), 
  #                  "\nSample Mean:", round(emp_mean, 3), " Var:", round(emp_vari, 3)), 
  #      x="Number of Events", y="Density")
  labs(title=paste(name, " - T=", t, 
                   "\nAsy Mean:", round(asy_mean, 3), " Var:", round(asy_var, 3), 
                   "\nSample Mean:", round(emp_mean, 3), " Var:", round(emp_vari, 3)), 
       x="Number of Events", y="Density")
  return (plt)
}

num_events_immig <- function(mu, alpha, beta, t){
  N_i = rpois(1, lambda = (mu * t))
  D_i = rpois(N_i, lambda = (alpha / beta))
  
  D_is = 0
  prev_num_childern = sum(D_i)
  while (prev_num_childern > 0){
    di = sum(rpois(prev_num_childern, lambda = (alpha / beta)))
    prev_num_childern = di
    D_is = D_is + di
  }
  
  return (N_i + sum(D_i) + D_is)
}

num_events_reg_hawkes <- function(mu, alpha, beta, t){
  a  = simulateHawkes(mu, alpha, beta, t)
  return (length(a[[1]]))
}

n_rep = 100000
# t = 200
mu = 0.5
main_alpha = 0.6
main_beta = 0.8
hawkes_n = rep(0, n_rep)
num_expected = (mu * t) / (1 - main_alpha / main_beta)

pdf("immigrant-model-hawkes-asy.pdf")
for (t in c(1, 2, 5, 10, 20, 35, 50, 60, 85, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500)){
# for (t in c(5, 10)){
  immigs = rep(0, n_rep)
  reg_hawkes = rep(0, n_rep)
  print(t)
  for (i in 1: n_rep){
    immigs[i] = num_events_immig(mu, main_alpha, main_beta, t)
    reg_hawkes[i] = num_events_reg_hawkes(mu, main_alpha, main_beta, t)
  }
  
  plt_immig <- plot_event_hist_log(immigs, "Immigrant", mu, main_alpha, main_beta, t)
  plt_reg <- plot_event_hist_log(reg_hawkes, "Regular Hawkes", mu, main_alpha, main_beta, t)
  
  grid.arrange(plt_immig, plt_reg, ncol=1)
}
dev.off()
