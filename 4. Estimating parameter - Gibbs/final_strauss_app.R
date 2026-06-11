# Loading packages -------------------
# In order to install keras, first run install.packages("keras") and then install_keras().  
install.packages("keras")
install.packages("tensorflow")
library(reticulate)
library(keras)
library(tensorflow)
install_keras()
install_tensorflow()

library(ads)
library(spatstat)
#library(tidyverse)
library(grid)
library(purrr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(GET)
library(pbmcapply)
library(patchwork)
library(keras)
library(parallel)

ncores <- ifelse(detectCores() == 1, 1, detectCores() - 1)

# Function for simulating Strauss processes --------------------------------------
rStrauss_rmh <- function(beta, gamma, R, W = owin(), ...){
  model <- list(cif = "strauss", par = list(beta = beta, gamma = gamma, r = R), w = W)
  X  <- rmh(model = model, start = list(n.start = 0), verbose = FALSE, p = 0, ...)
  as.ppp(X)
}

# Bivariate cells data ----------------------------------------------------------
df <- read.csv("fotoreceptor.csv", sep = ",")
head(df)
cell <- ppp(
  x = df$x,
  y = df$y,
  window = owin(
    xrange = range(df$x),
    yrange = range(df$y))
)
cell
plot(cell)
plot(
  df$x, df$y,
  pch = 19, cex = 0.6, col = "black",
  xlab = "x (µm)", ylab = "y (µm)"
)
setEPS()
postscript("plot_cell.eps")
plot(
  df$x, df$y,
  pch = 19, cex = 0.6, col = "blue",
  xlab = "x (µm)", ylab = "y (µm)"
)
dev.off()
X_obs <- cell

p1 <- ggplot(data.frame(x = X_obs$x, y = X_obs$y), aes(x, y)) +
  geom_point(size = 1) +
  xlim(X_obs$window$xrange) +
  ylim(X_obs$window$yrange) + 
  coord_fixed() +
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank()) +
  geom_segment(aes(x = X_obs$window$xrange[1], xend = X_obs$window$xrange[2],
                   y = X_obs$window$yrange[1]), yend = X_obs$window$yrange[1]) +
  geom_segment(aes(x = X_obs$window$xrange[1], xend = X_obs$window$xrange[2],
                   y = X_obs$window$yrange[2]), yend = X_obs$window$yrange[2]) +
  geom_segment(aes(x = X_obs$window$xrange[1], xend = X_obs$window$xrange[1],
                   y = X_obs$window$yrange[1]), yend = X_obs$window$yrange[2]) +
  geom_segment(aes(x = X_obs$window$xrange[2], xend = X_obs$window$xrange[2],
                   y = X_obs$window$yrange[1]), yend = X_obs$window$yrange[2])

W <- X_obs$window

env_pois <- envelope(X_obs, Lest, nsim = 2499, transform = expression(. - r), savefuns = TRUE)
p2 <- plot(global_envelope_test(env_pois)) +
  theme_bw() +
  theme(plot.title = element_blank(),
        legend.position = "none",
        panel.grid = element_blank()) +
  scale_x_continuous(expand = expansion(0))
p2

p1 + p2 

setEPS()
postscript("plot_cell+L.eps")
plot(p1 + p2)
dev.off()

# Testset ----------------------------------------------------
ntest <- 5000

beta <- runif(ntest, 0.001, 0.01)
gamma <- runif(ntest, 0.001, 1)
R <- runif(ntest, 1, 50)

Strauss_sims <- pbmcmapply(function(beta, gamma, R){
  rStrauss_rmh(beta = beta, gamma = gamma, R = R, W = W, nrep = 100000)
}, beta, gamma, R, SIMPLIFY = FALSE, mc.cores = ncores)

n_points <- sapply(Strauss_sims, spatstat.geom::npoints)
head(n_points)
summary(n_points)
hist(n_points, breaks = 30, col = "skyblue", main = "Number of points per simulation")
sum(n_points == 0)

Dataexample_test <- tibble(beta = beta, gamma = gamma, R = R, pp = Strauss_sims) %>% 
  mutate(L = pbmclapply(pp, function(ppp){
    L <- Lest(ppp, correction = "best")
    L$iso - L$r
  }, mc.cores = ncores),
  N = purrr::map_int(pp, ~ spatstat.geom::npoints(.x))) 

nsim = 5000
env_L <- envelope(X_obs, Lest, simulate = Dataexample_test$pp, nsim = 5000, savefuns = TRUE,
                  transform = expression(. - r))
GET_L <- global_envelope_test(env_L)
p2 <- plot(GET_L) +
  theme_bw() +
  theme(plot.title = element_blank(),
        legend.position = "none",
        panel.grid = element_blank()) +
  scale_x_continuous(expand = expansion(0))
p2

# Training data ----------------------------------
t_simtrain_start <- Sys.time()
ntrain <- 40000

beta <- runif(ntrain, 0.001, 0.01)
gamma <- runif(ntrain, 0.001, 1)
R <- runif(ntrain, 1, 50)

Strauss_sims <- pbmcmapply(function(beta, gamma, R){
  rStrauss_rmh(beta = beta, gamma = gamma, R = R, W = W, nrep = 100000)
}, beta, gamma, R, SIMPLIFY = FALSE, mc.cores = ncores)

#n_points <- sapply(Strauss_sims, spatstat.geom::npoints)
#head(n_points)
#summary(n_points)
#hist(n_points, breaks = 30, col = "skyblue", main = "Number of points per simulation")
#sum(n_points == 0)

Dataexample_Strauss <- tibble(beta = beta, gamma = gamma, R = R, pp = Strauss_sims) %>% 
  mutate(L = pbmclapply(pp, function(ppp){
    L <- Lest(ppp, correction = "best")
    L$iso - L$r
  }, mc.cores = ncores),
  N = vapply(.data$pp, function(z) z$n, integer(1))
  )
t_simtrain_end <- Sys.time()
time_simtraining <- difftime(t_simtrain_end, t_simtrain_start, units = "secs")
cat("Simulation training dataset time:", round(as.numeric(time_simtraining), 2), "seconds\n")

# Comparing observed point pattern to training data -----------------
p1 <- ggplot(mapping = aes(x = Dataexample_Strauss$N)) +
  geom_histogram(breaks = seq(min(Dataexample_Strauss$N), max(Dataexample_Strauss$N), length.out = 20),
                 color = "white", fill = "grey80") +
  geom_vline(xintercept = npoints(X_obs), color = "black", linetype = "dashed") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        plot.margin = margin(r = 1)) +
  ylab("Count") +
  xlab("Number of points") +
  scale_x_continuous(expand = expansion(0.01)) +
  scale_y_continuous(expand = expansion(0.01))
p1

nsim = 40000
env_L <- envelope(X_obs, Lest, simulate = Dataexample_Strauss$pp, nsim = 40000, savefuns = TRUE,
                  transform = expression(. - r))
GET_L <- global_envelope_test(env_L)
p2 <- plot(GET_L) +
  theme_bw() +
  theme(plot.title = element_blank(),
        legend.position = "none",
        panel.grid = element_blank()) +
  scale_x_continuous(expand = expansion(0))
p2

p1 + p2

setEPS()
postscript("plot_Ltest.eps")
plot(p1 + p2)
postscript("hist_test.eps")
plot(p1)
postscript("env_testset")
plot(p2)
dev.off()

# Training neural network -------------------------------
## Prepare training input
library(reticulate)
m_L <- mean(unlist(Dataexample_Strauss$L))
std_L <- sd(unlist(Dataexample_Strauss$L))
train_L <- lapply(Dataexample_Strauss$L, function(L){(L - m_L) / std_L})
train_L <- array_reshape(train_L, c(nrow(Dataexample_Strauss), length(train_L[[1]]), 1))

train_N <- select(Dataexample_Strauss, N)
m_N <- apply(train_N, 2, mean)
std_N <- apply(train_N, 2, sd)
train_N <- scale(as.matrix(train_N), center = m_N, scale = std_N)

train_par <- as.matrix(select(Dataexample_Strauss, beta:R))
m_par <- apply(train_par, 2, mean)
std_par <- apply(train_par, 2, sd)
train_par <- scale(train_par, center = m_par, scale = std_par)

## Prepare test input
test_L <- lapply(Dataexample_test$L, function(L){(L - m_L) / std_L})
test_L <- array_reshape(test_L, c(nrow(Dataexample_test), length(test_L[[1]]), 1))

test_N <- scale(as.matrix(select(Dataexample_test, N)), center = m_N, scale = std_N)

test_par <- scale(as.matrix(select(Dataexample_test, beta:R)), center = m_par, scale = std_par)

## Prepare observed input
L <- Lest(X_obs, correction = "best")
obs_L <- ((L$iso - L$r) - m_L) / std_L
obs_L <- array_reshape(obs_L, c(1, length(obs_L), 1))

obs_N <- (npoints(X_obs) - m_N) / std_N
obs_N <- as.matrix((npoints(X_obs) - m_N) / std_N)

## Build neural network
main_input <- layer_input(shape = dim(train_L)[-1], name = 'main_input')

conv_out <- main_input %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu",
                input_shape = dim(train_L)[-1]) %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_flatten()

auxiliary_input <- layer_input(shape = c(1), name = 'aux_input')

main_output <- layer_concatenate(c(conv_out, auxiliary_input)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = ncol(train_par), activation = "linear", name = "main_output")

model <- keras_model(
  inputs = c(main_input, auxiliary_input),
  outputs = c(main_output)
)

model$compile(loss = "mse", optimizer = "adam")

## Train neural network (TIMED) ----------------------------------------
t_train_start <- Sys.time()

history <- model$fit(
  x = list(train_L, train_N),
  y = train_par,
  epochs = 20L,
  batch_size = 100L,
  validation_data = list(list(test_L, test_N), test_par)
)

t_train_end <- Sys.time()
time_training <- difftime(t_train_end, t_train_start, units = "secs")
cat("Neural network training time:", round(as.numeric(time_training), 2), "seconds\n")

## Loss plot
df <- as.data.frame(history$history)
df$epoch <- seq_len(nrow(df))
print(names(df))

loss_df <- data.frame(
  epoch = rep(df$epoch, 2),
  value = c(df$loss, df$val_loss),
  type = rep(c("Training", "Validation"), each = nrow(df))
)

ggplot(loss_df, aes(x = epoch, y = value, color = type)) +
  geom_line(size = 0.5) +
  geom_point(size = 0.8) +
  theme_bw() +
  labs(
    x = "Epoch",
    y = "Loss",
    color = ""
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 7),
    legend.position = "right"
  )

# Estimates obtained for the point patterns in the test set (TIMED) ------------
t_test_start <- Sys.time()

estimates_test <- model$predict(list(test_L, test_N))
estimates_test <- sweep(estimates_test, 2, std_par, "*")
estimates_test <- sweep(estimates_test, 2, m_par, "+")
colnames(estimates_test) <- colnames(train_par)

t_test_end <- Sys.time()
time_test <- difftime(t_test_end, t_test_start, units = "secs")
cat("Neural network test-set prediction time (", nrow(test_par), "patterns):",
    round(as.numeric(time_test), 4), "seconds\n")

# Plots 
test_par_plot <- test_par %>% 
  sweep(2, std_par, "*") %>% 
  sweep(2, m_par, "+")

beta <- runif(ntrain, 0.001, 0.01)
gamma <- runif(ntrain, 0.001, 1)
R <- runif(ntrain, 1, 50)

library(ggplot2)
(p1 <- ggplot() +
    geom_point(aes(test_par_plot[, "beta"], estimates_test[, "beta"]), alpha = 0.2, size = 0.5, stroke = 0) +
    theme_bw() +
    theme(panel.grid = element_blank(),
          plot.margin = margin(r = 3, unit = "pt"),
          axis.title = element_blank(),
          axis.text = element_text(size = 8),
          plot.title = element_text(hjust = 0.5, size = 8)) +
    ggtitle(expression(beta)) +
    ylim(0.001, 0.01)+
    geom_segment(aes(x=0.001, xend = 0.01, y = 0.001, yend = 0.01), color = "grey50") +
    scale_x_continuous(expand = expansion(0.01)))

(p2 <- ggplot() +
    geom_point(aes(test_par_plot[, "gamma"], estimates_test[, "gamma"]), alpha = 0.2, size = 0.5, stroke = 0) +
    theme_bw() +
    theme(panel.grid = element_blank(),
          plot.margin = margin(r = 3, unit = "pt"),
          axis.title = element_blank(),
          axis.text = element_text(size = 8),
          plot.title = element_text(hjust = 0.5, size = 8)) +
    ylim(0.001, 1) +
    ggtitle(expression(gamma)) +
    geom_segment(aes(x=0.001, xend = 1, y = 0.001, yend = 1), color = "grey50") +
    scale_x_continuous(expand = expansion(0.01)))

(p3 <- ggplot() +
    geom_point(aes(test_par_plot[, "R"], estimates_test[, "R"]), alpha = 0.2, size = 0.5, stroke = 0) +
    geom_segment(aes(x=1, xend = 50, y = 1, yend = 50), color = "grey50") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          plot.margin = margin(r = 3, unit = "pt"),
          axis.title = element_blank(),
          axis.text = element_text(size = 8),
          plot.title = element_text(hjust = 0.5, size = 8)) +
    ggtitle(expression(R)) +
    ylim(1, 50) +
    scale_x_continuous(expand = expansion(0.01)))

gridExtra::grid.arrange(patchwork::patchworkGrob(p1 + p2 + p3), left = "Estimate", bottom = "True")

setEPS()
postscript("plot_est vs true.eps")
plot(gridExtra::grid.arrange(patchwork::patchworkGrob(p1 + p2 + p3), left = "Estimate", bottom = "True"))
dev.off()

#error_data <- data.frame(
#  beta_true = test_par_plot[, "beta"],
#  gamma_true = test_par_plot[, "gamma"],
#  R_true = test_par_plot[, "R"],
#  beta_est = estimates_test[, "beta"],
#  gamma_est = estimates_test[, "gamma"],
#  R_est = estimates_test[, "R"]
#) %>%
#  mutate(
#    beta_error = beta_est - beta_true,
#    gamma_error = gamma_est - gamma_true,
#    R_error = R_est - R_true
#  )

#make_boxplot <- function(data, true_col, error_col, label) {
#  ggplot(data, aes(x = cut(!!sym(true_col), 5), y = !!sym(error_col))) +
#    geom_boxplot(fill = "grey60", color = "black", outlier.size = 0.8, width = 0.6) +
#    theme_bw() +
#    theme(
#      panel.grid = element_blank(),
#      axis.title = element_text(size = 9),
#      axis.text = element_text(size = 8),
#      axis.text.x = element_text(angle = 45, hjust = 1),
#      plot.title = element_text(hjust = 0.5, size = 9)
#    ) +
#    ggtitle(label) +
#    ylab("Error") +
#    xlab("True parameter") +
#    geom_hline(yintercept = 0, color = "black", size = 0.4)
#}

#p1 <- make_boxplot(error_data, "beta_true", "beta_error", expression(beta))
#p2 <- make_boxplot(error_data, "gamma_true", "gamma_error", expression(gamma))
#p3 <- make_boxplot(error_data, "R_true", "R_error", expression(R))

#p1 + p2 + p3

#error_beta <- estimates_test[, "beta"] - test_par_plot[, "beta"]
#error_gamma <- estimates_test[, "gamma"] - test_par_plot[, "gamma"]
#error_R <- estimates_test[, "R"] - test_par_plot[, "R"]

#p1 <- ggplot() +
#  geom_point(aes( test_par_plot[, "beta"], error_gamma), alpha = 0.2, size = 0.5, stroke = 0) +
#  geom_hline(yintercept = 0, color = "grey50") +
#  theme_bw() +
#  theme(panel.grid = element_blank(),
#        plot.margin = margin(r = 8, unit = "pt"),
#        plot.title = element_text(hjust = 0.5, size = 8),
#        axis.title = element_text(size = 8),
#        axis.text = element_text(size = 8)) +
#  scale_x_continuous(expand = expansion(0.01)) +
#  xlab(expression(beta)) + 
#  ylab(expression(hat(gamma) - gamma)) +
#  ylim(-1, 1)

#p2 <- ggplot() +
#  geom_point(aes( test_par_plot[, "gamma"], error_beta), alpha = 0.2, size = 0.5, stroke = 0) +
#  geom_hline(yintercept = 0, color = "grey50") +
#  theme_bw() +
#  theme(panel.grid = element_blank(),
#        plot.margin = margin(r = 8, unit = "pt"),
#        plot.title = element_text(hjust = 0.5, size = 8),
#        axis.title = element_text(size = 8),
#        axis.text = element_text(size = 8)) +
#  scale_x_continuous(expand = expansion(0.01)) +
#  xlab(expression(gamma)) + 
#  ylab(expression(hat(beta) - beta)) +
#  ylim(-0.01, 0.01)

#p3 <- ggplot() +
#  geom_point(aes( test_par_plot[, "beta"], error_R), alpha = 0.2, size = 0.5, stroke = 0) +
#  geom_hline(yintercept = 0, color = "grey50") +
#  theme_bw() +
#  theme(panel.grid = element_blank(),
#        plot.margin = margin(r = 8, unit = "pt"),
#        plot.title = element_text(hjust = 0.5, size = 8),
#        axis.title = element_text(size = 8),
#        axis.text = element_text(size = 8)) +
#  scale_x_continuous(expand = expansion(0.01)) +
#  xlab(expression(beta)) + 
#  ylab(expression(hat(R) - R)) +
#  ylim(-50, 50)

#p4 <- ggplot() +
#  geom_point(aes( test_par_plot[, "R"], error_beta), alpha = 0.2, size = 0.5, stroke = 0) +
#  geom_hline(yintercept = 0, color = "grey50") +
#  theme_bw() +
#  theme(panel.grid = element_blank(),
#        plot.margin = margin(r = 8, unit = "pt"),
#        plot.title = element_text(hjust = 0.5, size = 8),
#        axis.title = element_text(size = 8),
#        axis.text = element_text(size = 8)) +
#  scale_x_continuous(expand = expansion(0.01)) +
#  xlab(expression(R)) + 
#  ylab(expression(hat(beta) - beta)) +
#  ylim(-0.01, 0.01)

#p5 <- ggplot() +
#  geom_point(aes( test_par_plot[, "gamma"], error_R), alpha = 0.2, size = 0.5, stroke = 0) +
#  geom_hline(yintercept = 0, color = "grey50") +
#  theme_bw() +
#  theme(panel.grid = element_blank(),
#        plot.margin = margin(r = 8, unit = "pt"),
#        plot.title = element_text(hjust = 0.5, size = 12),
#        axis.title = element_text(size = 8),
#        axis.text = element_text(size = 8)) +
#  scale_x_continuous(expand = expansion(0.01)) +
#  xlab(expression(gamma)) + 
#  ylab(expression(hat(R) - R)) +
#  ylim(-50, 50)

#p6 <- ggplot() +
#  geom_point(aes( test_par_plot[, "R"], error_gamma), alpha = 0.2, size = 0.5, stroke = 0) +
#  geom_hline(yintercept = 0, color = "grey50") +
#  theme_bw() +
#  theme(panel.grid = element_blank(),
#        plot.margin = margin(r = 8, unit = "pt"),
#        plot.title = element_text(hjust = 0.5, size = 8),
#        axis.title = element_text(size = 8),
#        axis.text = element_text(size = 8)) +
#  scale_x_continuous(expand = expansion(0.01)) +
#  xlab(expression(R)) + 
#  ylab(expression(hat(gamma) - gamma)) +
#  ylim(-1, 1)

#(p2 + p4) /
#  (p1 + p6) /
#  (p3 + p5)

# Estimates obtained for the observed point pattern ----------------------------
t_obs_start <- Sys.time()
estimates_obs <- model$predict(list(obs_L, obs_N))
estimates_obs <- sweep(estimates_obs, 2, std_par, "*")
estimates_obs <- sweep(estimates_obs, 2, m_par, "+")
colnames(estimates_obs) <- colnames(train_par)
estimates_obs
t_obs_end <- Sys.time()
time_obs <- difftime(t_obs_end, t_obs_start, units = "secs")
cat("NN estimation time:", round(as.numeric(time_obs), 2), "seconds\n")


# Envelope for validating fitted model -----------------------------------------
library(pbmcapply)
sim_model <- pbmclapply(
  1:2499,
  function(i) {
    rStrauss_rmh(
      beta = estimates_obs[, "beta"],
      gamma = estimates_obs[, "gamma"],
      R = estimates_obs[, "R"],
      W = W,
      nrep = 100000
    )
  },
  mc.cores = ncores
)

env <- envelope(X_obs, simulate = sim_model, savefuns = TRUE, nsim = 2499)
dclf.test(env)

# Classic estimation ---------------------------------------------------
library(spatstat)

t_classic_start <- Sys.time()

Xu <- X_obs

# Estimate R via profile pseudo-likelihood
nn <- nndist(Xu)
summary(nn)
rr <- data.frame(r = seq(min(nn), max(nn), by=0.001))
R_classic_start <- Sys.time()
ps <- profilepl(rr, Strauss, Xu, emend = TRUE)
R_classic_end <- Sys.time()
R_classic <- difftime(R_classic_end, R_classic_start, units = "secs")
cat("R Classic estimation time:", round(as.numeric(R_classic), 2), "seconds\n")
R_hat <- rr$r[ps$iopt]

# Estimate beta and gamma given R_hat
fit0 <- ppm(Xu ~ 1, Strauss(R_hat))
fit0
co <- coef(fit0)
beta_hat  <- exp(co[1])
gamma_hat <- ifelse(length(co) >= 2, exp(co[2]), 1)

t_classic_end <- Sys.time()
time_classic <- difftime(t_classic_end, t_classic_start, units = "secs")
cat("Classic estimation time:", round(as.numeric(time_classic), 2), "seconds\n")

cat("Classic estimates:\n")
cat("  R_hat     =", R_hat, "\n")
cat("  beta_hat  =", beta_hat, "\n")
cat("  gamma_hat =", gamma_hat, "\n")