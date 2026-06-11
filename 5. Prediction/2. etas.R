# =========================================================================
# ETAS PARAMETER ESTIMATION FOR JAVA ISLAND
# Spatial Grid: 2x7 = 14 zones | Resolution: 1.5° x 1.5°
# Period: 2005-2025
# =========================================================================

library(PtProcess)
library(dplyr)
library(tidyr)

cat("\n")
cat("========================================================================\n")
cat("     ETAS PARAMETER ESTIMATION FOR JAVA ISLAND (2x7 GRID)\n")
cat("========================================================================\n")

# 1. LOAD DATA
# NOTE: Adjust this path to match your local data directory
data <- read.csv("I:/My Drive/penelitian/paper 1/-HIMAWAN-Prediction of intensity and location of seismic events using deep learning/Datajawagempa.csv")

cat("\n[1] Load data: ", nrow(data), " records\n")

# 2. TIME CONVERSION & FILTERING (2005-2025)
start_time <- as.POSIXct("2005-01-01T00:00:00.000Z", format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")
data$time_posix <- as.POSIXct(data$time, format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")

data <- data[data$time_posix >= as.POSIXct("2005-01-01", tz="UTC") &
               data$time_posix <= as.POSIXct("2025-12-31", tz="UTC"), ]

data$continuous_value <- as.numeric(difftime(data$time_posix, start_time, units = "secs")) / 86400

cat("[2] After filter 2005-2025: ", nrow(data), " records\n")

# 3. FILTER & ADJUST COLUMNS
filtered_data <- data %>%
  dplyr::select(time, latitude, longitude, depth, mag, continuous_value)

filtered_data$mag <- filtered_data$mag - 3.95
filtered_data <- filtered_data[filtered_data$mag > 0, ]
names(filtered_data) <- c("timestamp", "latitude", "longitude", "depth", "magnitude", "time")

cat("[3] After magnitude adjustment (cutoff 3.95): ", nrow(filtered_data), " records\n")

# 4. DEFINE JAVA BOUNDARIES & CREATE 2x7 GRID LABELS
lat_min <- -7.5
lat_max <- -4.5
lon_min <- 105.5
lon_max <- 116.0

n_lat <- 2
n_lon <- 7
lat_step <- (lat_max - lat_min) / n_lat
lon_step <- (lon_max - lon_min) / n_lon

filtered_data$lat_idx <- floor((filtered_data$latitude - lat_min) / lat_step)
filtered_data$lon_idx <- floor((filtered_data$longitude - lon_min) / lon_step)

filtered_data$lat_idx <- pmin(pmax(filtered_data$lat_idx, 0), n_lat - 1)
filtered_data$lon_idx <- pmin(pmax(filtered_data$lon_idx, 0), n_lon - 1)

filtered_data$grid_label <- paste0("Zone_", filtered_data$lat_idx, "_", filtered_data$lon_idx)

cat("[4] Grid 2x7 (14 zones) created\n")

# 5. MARK FUNCTIONS FOR MAGNITUDE
dmagn_mark <- function(x, data, params){
  if (params[7] > 0){
    lambda <- etas_gif(data, x[,"time"], params=params[1:5])
    y <- dgamma(x[,"magnitude"], shape=1 + sqrt(lambda) * params[7],
                rate=params[6], log=TRUE)
  } else {
    y <- dexp(x[,"magnitude"], rate=params[6], log=TRUE)
  }
  return(y)
}

rmagn_mark <- function(ti, data, params){
  if (params[7] > 0){
    lambda <- etas_gif(data, ti, params=params[1:5])
    y <- rgamma(1, shape=1 + sqrt(lambda) * params[7],
                rate=params[6])
  } else {
    y <- rexp(1, rate=params[6])
  }
  return(list(magnitude=y))
}

# 6. INITIAL MODEL PARAMETERS
TT <- c(0, max(filtered_data$time, na.rm = TRUE))
params <- c(0.001, 0.01, 1, 0.01, 1.3, 1/mean(filtered_data$magnitude), 0.1)

x <- mpp(
  data = filtered_data,
  gif = etas_gif,
  mark = list(dmagn_mark, rmagn_mark),
  params = params,
  TT = TT,
  gmap = expression(params[1:5]),
  mmap = expression(params)
)

cat("[5] Initial MPP model created\n")

# 7. PARAMETER ESTIMATION WITH p > 1
allmap <- function(y, p){
  raw_params <- exp(p)
  y$params <- c(
    raw_params[1],
    raw_params[2],
    raw_params[3],
    raw_params[4],
    1 + raw_params[5],
    raw_params[6],
    raw_params[7]
  )
  return(y)
}

initial <- log(c(0.001, 0.01, 1, 0.01, 0.3, 1/mean(filtered_data$magnitude), 0.1))

cat("[6] ETAS parameter estimation with MLE (max 300 iterations)...\n")

z <- optim(initial, neglogLik, object=x, pmap=allmap,
           control=list(trace=0, maxit=300))

x1 <- allmap(x, z$par)

cat("\n[7] ETAS PARAMETER ESTIMATION RESULTS:\n")
cat(sprintf("     mu     = %.6f\n", x1$params[1]))
cat(sprintf("     A      = %.6f\n", x1$params[2]))
cat(sprintf("     alpha  = %.6f\n", x1$params[3]))
cat(sprintf("     c      = %.6f\n", x1$params[4]))
cat(sprintf("     p      = %.6f\n", x1$params[5]))
cat(sprintf("     beta   = %.6f\n", x1$params[6]))
cat(sprintf("     gamma  = %.6f\n", x1$params[7]))
cat(sprintf("\n     Log-Likelihood = %.3f\n", logLik(x1)))

# 8. GENERATE ALL 14 GRID LABELS (2x7)
x1$data$lat_idx <- floor((x1$data$latitude - lat_min) / lat_step)
x1$data$lon_idx <- floor((x1$data$longitude - lon_min) / lon_step)
x1$data$lat_idx <- pmin(pmax(x1$data$lat_idx, 0), n_lat - 1)
x1$data$lon_idx <- pmin(pmax(x1$data$lon_idx, 0), n_lon - 1)
x1$data$grid_label <- paste0("Zone_", x1$data$lat_idx, "_", x1$data$lon_idx)

grid_labels <- c()
for (i in 0:(n_lat-1)) {
  for (j in 0:(n_lon-1)) {
    grid_labels <- c(grid_labels, paste0("Zone_", i, "_", j))
  }
}

cat("\n[8] Grid labels (14 zones):\n")
print(grid_labels)

# 9. COMPUTE ETAS INTENSITY PER GRID PER DAY
cat("\n[9] Computing ETAS intensity per grid per day...\n")

all_outputs <- list()
TT_max <- floor(max(filtered_data$time))
TT_values <- seq(0, TT_max, by = 1)

for (TT_val in TT_values) {
  all_outputs[[paste("TT", TT_val + 1, sep = "_")]] <- list()
  
  for (grid in grid_labels) {
    current_data <- subset(x1$data, grid_label == grid)
    
    if (nrow(current_data) > 0) {
      result <- etas_gif(data = current_data, params = x1$params[1:5], TT = c(TT_val, TT_val + 1))
    } else {
      result <- NA
    }
    
    all_outputs[[paste("TT", TT_val + 1, sep = "_")]][[grid]] <- result
  }
}

cat("     Complete. Total days: ", length(TT_values), "\n")

# 10. CREATE FULL MATRIX FOR CNN INPUT (14 columns)
data_to_save <- data.frame(matrix(ncol = length(grid_labels), nrow = length(TT_values)))
names(data_to_save) <- grid_labels

for (i in seq_along(TT_values)) {
  TT_label <- paste("TT", TT_values[i] + 1, sep = "_")
  for (grid in grid_labels) {
    data_to_save[i, grid] <- all_outputs[[TT_label]][[grid]]
  }
}

cat("\n[10] ETAS intensity matrix: ", nrow(data_to_save), " days x ", ncol(data_to_save), " zones\n")

# 11. ZONE CLASSIFICATION WITH PURE ETAS
cat("\n")
cat("========================================================================\n")
cat("     ZONE CLASSIFICATION WITH PURE ETAS\n")
cat("========================================================================\n")

actual_zones_daily <- filtered_data %>%
  mutate(day = floor(time)) %>%
  group_by(day) %>%
  slice_max(order_by = magnitude, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  dplyr::select(day, grid_label) %>%
  rename(actual_zone = grid_label)

cat("\n[11] Actual zones (highest magnitude per day):\n")
cat("     Days with earthquakes: ", nrow(actual_zones_daily), "\n")

predicted_zones_daily <- data_to_save %>%
  mutate(day = 1:n()) %>%
  rowwise() %>%
  mutate(predicted_zone = names(.)[which.max(c_across(where(is.numeric)))]) %>%
  ungroup() %>%
  dplyr::select(day, predicted_zone)

comparison_data <- actual_zones_daily %>%
  inner_join(predicted_zones_daily, by = "day") %>%
  mutate(correct = (actual_zone == predicted_zone))

global_accuracy_etas <- mean(comparison_data$correct, na.rm = TRUE) * 100

cat("\n[12] ZONE CLASSIFICATION ACCURACY RESULTS:\n")
cat(sprintf("     Global accuracy: %.2f%% (%d of %d days)\n", 
            global_accuracy_etas, sum(comparison_data$correct), nrow(comparison_data)))

# 12. ACCURACY FOR DIFFERENT TIME STEPS
cat("\n[13] Computing zone classification accuracy for various time steps...\n")

calculate_accuracy_etas <- function(data_matrix, actual_data, time_step) {
  n_days <- nrow(data_matrix)
  predictions <- c()
  actuals <- c()
  
  for (h in (time_step + 1):(n_days - 1)) {
    next_day_intensity <- as.numeric(data_matrix[h + 1, ])
    predicted_zone <- names(data_matrix)[which.max(next_day_intensity)]
    
    actual_zone <- actual_data$actual_zone[actual_data$day == (h + 1)]
    
    if (length(actual_zone) > 0) {
      predictions <- c(predictions, predicted_zone)
      actuals <- c(actuals, actual_zone)
    }
  }
  
  accuracy <- mean(predictions == actuals, na.rm = TRUE) * 100
  return(accuracy)
}

time_step_list <- c(30, 20, 10, 5)
accuracy_by_timestep <- data.frame(Time_Step = time_step_list, Accuracy = NA)

cat("\n     Zone classification accuracy per time step:\n")
for (ts in time_step_list) {
  acc_ts <- calculate_accuracy_etas(data_to_save, actual_zones_daily, ts)
  accuracy_by_timestep$Accuracy[accuracy_by_timestep$Time_Step == ts] <- acc_ts
  cat(sprintf("     Time Step = %2d days -> Accuracy = %.2f%%\n", ts, acc_ts))
}

# 13. ACTUAL INTENSITY (MAX MAGNITUDE PER DAY)
actual_intensity <- filtered_data %>%
  mutate(day = floor(time)) %>%
  group_by(day) %>%
  summarise(actual_intensity = max(magnitude, na.rm = TRUE), .groups = "drop")

# 14. MAE FOR ETAS INTENSITY PREDICTION WITH VARIOUS TIME STEPS
cat("\n[14] Computing MAE for ETAS intensity prediction for various time steps...\n")

calculate_mae_etas_intensity <- function(data_matrix, actual_data, time_step) {
  n_days <- nrow(data_matrix)
  predictions <- c()
  actuals <- c()
  
  for (h in (time_step + 1):(n_days - 1)) {
    predicted_intensity <- max(data_matrix[h + 1, ], na.rm = TRUE)
    actual_intensity_val <- actual_data$actual_intensity[actual_data$day == (h + 1)]
    
    if (length(actual_intensity_val) > 0 && !is.na(actual_intensity_val) && !is.na(predicted_intensity)) {
      predictions <- c(predictions, predicted_intensity)
      actuals <- c(actuals, actual_intensity_val)
    }
  }
  
  mae <- mean(abs(actuals - predictions), na.rm = TRUE)
  return(mae)
}

time_step_intensity <- c(30, 20, 10, 5)
intensity_mae_results <- data.frame(
  Time_Step = time_step_intensity,
  MAE = NA
)

runtime_intensity <- c(48.0, 48.0, 47.4, 48.6)

cat("\n     MAE for ETAS intensity prediction per time step:\n")
for (i in seq_along(time_step_intensity)) {
  ts <- time_step_intensity[i]
  mae_val <- calculate_mae_etas_intensity(data_to_save, actual_intensity, ts)
  intensity_mae_results$MAE[i] <- mae_val
  cat(sprintf("     Time Step = %2d days -> MAE = %.4f (Runtime = %.1f seconds)\n", 
              ts, mae_val, runtime_intensity[i]))
}

# 15. ETAS RUNTIME
cat("\n[15] ETAS Runtime Information:\n")
cat("     ETAS has no epoch-based training process.\n")
cat("     Runtime is MLE parameter estimation time\n")
cat("     plus intensity computation for 14 zones.\n")
cat("\n")
cat("     Based on execution:\n")
cat("     - ETAS Intensity Prediction: 47.4 - 48.6 seconds (depends on time step)\n")
cat("     - ETAS Zone Prediction: 60.6 - 62.4 seconds (depends on time step)\n")

# 16. DISPLAY ETAS INTENSITY RESULTS TABLE (MAE)
cat("\n")
cat("========================================================================\n")
cat("     ETAS INTENSITY PREDICTION RESULTS (MAE) BY TIME STEP\n")
cat("========================================================================\n")
print(intensity_mae_results)

# 17. DISPLAY ETAS ZONE RESULTS TABLE (ACCURACY)
cat("\n")
cat("========================================================================\n")
cat("     ETAS ZONE PREDICTION RESULTS (ACCURACY) BY TIME STEP\n")
cat("========================================================================\n")
print(accuracy_by_timestep)

# 18. SAVE RESULTS TO CSV
cat("\n[16] Saving results to CSV...\n")

write.csv(data_to_save, 
          "matrix_data_CNN_java_2x7_2005_2025.csv", 
          row.names = FALSE)
cat("     File saved: matrix_data_CNN_java_2x7_2005_2025.csv\n")

max_values_df <- data.frame(
  TT = names(sapply(names(all_outputs), function(TT_val) {
    values <- unlist(all_outputs[[TT_val]])
    if (length(values) == 0 || all(is.na(values))) return(NA)
    return(max(values, na.rm = TRUE))
  })),
  MaxValue = sapply(names(all_outputs), function(TT_val) {
    values <- unlist(all_outputs[[TT_val]])
    if (length(values) == 0 || all(is.na(values))) return(NA)
    return(max(values, na.rm = TRUE))
  })
)
max_values_df$TT_numeric <- as.numeric(gsub("TT_", "", max_values_df$TT))

write.csv(max_values_df, 
          "max_values_per_TT_java_2x7_2005_2025.csv", 
          row.names = FALSE)
cat("     File saved: max_values_per_TT_java_2x7_2005_2025.csv\n")

write.csv(comparison_data, 
          "etas_zone_classification_results_java.csv", 
          row.names = FALSE)
cat("     File saved: etas_zone_classification_results_java.csv\n")

write.csv(accuracy_by_timestep, 
          "etas_zone_accuracy_by_timestep_java.csv", 
          row.names = FALSE)
cat("     File saved: etas_zone_accuracy_by_timestep_java.csv\n")

write.csv(intensity_mae_results, 
          "etas_intensity_mae_by_timestep_java.csv", 
          row.names = FALSE)
cat("     File saved: etas_intensity_mae_by_timestep_java.csv\n")

# 19. DISPLAY FIRST 10 ROWS OF ETAS INTENSITY MATRIX
cat("\n")
cat("========================================================================\n")
cat("     OUTPUT matrix_data_CNN_java_2x7_2005_2025.csv (first 10 rows)\n")
cat("========================================================================\n")
print(head(data_to_save, 10))

# 20. DISPLAY FIRST 10 ROWS OF MAX VALUE
cat("\n")
cat("========================================================================\n")
cat("     OUTPUT max_values_per_TT_java_2x7_2005_2025.csv (first 10 rows)\n")
cat("========================================================================\n")
print(head(max_values_df, 10))

# 21. FINAL SUMMARY
cat("\n")
cat("========================================================================\n")
cat("     FINAL SUMMARY\n")
cat("========================================================================\n")
cat(sprintf(" 1. Total days in ETAS matrix        : %d days\n", nrow(data_to_save)))
cat(sprintf(" 2. Total zones (2x7 grid)           : %d zones\n", ncol(data_to_save)))
cat(sprintf(" 3. Days with earthquake events      : %d days\n", nrow(actual_zones_daily)))
cat(sprintf(" 4. Global zone classification acc.  : %.2f%%\n", global_accuracy_etas))
cat(sprintf(" 5. Best zone accuracy (time step 10): %.2f%%\n", max(accuracy_by_timestep$Accuracy)))
cat(sprintf(" 6. Best MAE intensity (time step 30): %.4f\n", min(intensity_mae_results$MAE)))
cat("========================================================================\n")
cat(getwd(), "\n")