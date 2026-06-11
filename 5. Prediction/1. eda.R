# =====================================================
# Exploratory Data Analysis for Java Island
# Period: 2005–2025
# Spatial Grid: 2 × 7 (Square Grid 1.5° × 1.5°)
# =====================================================

library(ggplot2)
library(dplyr)
library(sf)
library(maps)
library(mapdata)

# =====================================================
# 1. Load Data (Adjust this path to match your local data directory)
# =====================================================
data <- read.csv(
  "I:/My Drive/penelitian/paper 1/-HIMAWAN-Prediction of intensity and location of seismic events using deep learning/Datajawagempa.csv"
)

# =====================================================
# 2. Time Processing (2005–2025)
# =====================================================
start_time <- as.POSIXct("2005-01-01T00:00:00Z", tz="UTC")
data$time_posix <- as.POSIXct(data$time, format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")

data <- data %>%
  filter(time_posix >= as.POSIXct("2005-01-01", tz="UTC"),
         time_posix <= as.POSIXct("2025-12-31", tz="UTC"))

data$continuous_value <- as.numeric(
  difftime(data$time_posix, start_time, units = "days")
)

# =====================================================
# 3. Variable Preparation
# =====================================================
filtered_data <- data %>%
  transmute(
    timestamp = time_posix,
    latitude,
    longitude,
    depth,
    magnitude = mag - 3.95,
    time = continuous_value
  ) %>%
  filter(magnitude > 0)

# =====================================================
# 4. Scientific Bounding Box – Java Island
# =====================================================
lat_min <- -8.83
lat_max <- -6.13
lon_min <- 105.20
lon_max <- 114.60

filtered_data <- filtered_data %>%
  filter(latitude  >= lat_min & latitude  <= lat_max,
         longitude >= lon_min & longitude <= lon_max)

# =====================================================
# 5. Square Grid Definition (1.5° × 1.5°)
# =====================================================
grid_size <- 1.5

n_lat <- ceiling((lat_max - lat_min) / grid_size)  # 2
n_lon <- ceiling((lon_max - lon_min) / grid_size)  # 7

filtered_data$lat_idx <- floor((filtered_data$latitude  - lat_min) / grid_size)
filtered_data$lon_idx <- floor((filtered_data$longitude - lon_min) / grid_size)

filtered_data$lat_idx <- pmin(pmax(filtered_data$lat_idx, 0), n_lat - 1)
filtered_data$lon_idx <- pmin(pmax(filtered_data$lon_idx, 0), n_lon - 1)

filtered_data$grid_label <- paste0(
  "Grid_", filtered_data$lat_idx + 1, "_", filtered_data$lon_idx + 1
)

# =====================================================
# 6. Magnitude Distribution
# =====================================================
ggplot(filtered_data, aes(x = magnitude)) +
  geom_histogram(binwidth = 0.1, fill = "steelblue", color = "black") +
  labs(
    title = "Magnitude Distribution of Earthquakes in Java Island (2005–2025)",
    x = "Magnitude (Mw)",
    y = "Frequency"
  ) +
  theme_minimal()

# =====================================================
# 7. Depth Distribution
# =====================================================
ggplot(filtered_data, aes(x = depth)) +
  geom_histogram(binwidth = 5, fill = "darkgreen", color = "black") +
  geom_vline(aes(xintercept = mean(depth, na.rm = TRUE)),
             color = "blue", linetype = "dashed") +
  geom_vline(aes(xintercept = median(depth, na.rm = TRUE)),
             color = "red", linetype = "dotted") +
  labs(
    title = "Depth Distribution of Earthquakes in Java Island (2005–2025)",
    x = "Depth (km)",
    y = "Frequency"
  ) +
  theme_minimal()

# =====================================================
# 8. Cumulative Number of Events
# =====================================================
filtered_data %>%
  mutate(year = format(as.Date(timestamp), "%Y")) %>%
  group_by(year) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(cumulative_count = cumsum(count)) %>%
  ggplot(aes(x = as.numeric(year), y = cumulative_count)) +
  geom_line(color = "red") +
  geom_point(color = "blue") +
  labs(
    title = "Cumulative Number of Earthquakes in Java Island (2005–2025)",
    x = "Year",
    y = "Cumulative Count"
  ) +
  theme_minimal()

# =====================================================
# 9. Gutenberg–Richter Law
# =====================================================
event_counts <- filtered_data %>%
  count(magnitude) %>%
  mutate(LogCount = log10(n))

ggplot(event_counts, aes(x = magnitude, y = LogCount)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  labs(
    title = "Log(Frequency) vs Magnitude (Gutenberg–Richter Law)",
    x = "Magnitude (Mw)",
    y = "Log₁₀(Event Count)"
  ) +
  theme_minimal()

# =====================================================
# 10. Map of Java Island with Square Grid
# =====================================================
map_data <- st_as_sf(
  map("worldHires",
      xlim = c(lon_min, lon_max),
      ylim = c(lat_min, lat_max),
      plot = FALSE,
      fill = TRUE)
)

ggplot() +
  geom_sf(data = map_data, fill = "lightgray", color = "black") +
  geom_hline(
    yintercept = seq(lat_min, lat_min + n_lat * grid_size, by = grid_size),
    color = "blue", linetype = "dashed"
  ) +
  geom_vline(
    xintercept = seq(lon_min, lon_min + n_lon * grid_size, by = grid_size),
    color = "red", linetype = "dashed"
  ) +
  coord_sf(
    xlim = c(lon_min, lon_min + n_lon * grid_size),
    ylim = c(lat_min, lat_min + n_lat * grid_size),
    expand = FALSE
  ) +
  labs(
    title = "Seismic Grid Map of Java Island (Square Grid 1.5° × 1.5°)",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal()