load("C:/Users/HP/Downloads/sumatra_earthquake.RData")
data

library(dplyr)
library(lubridate)

data_filtered <- data %>%
  mutate(
    time_parsed = ymd_hms(time),
    year = year(time_parsed)
  ) %>%
  filter(
    mag   >= 3,
    depth <  60,
    year  >= 2008,
    year  <= 2023
  ) %>%
  select(time, latitude, longitude)

head(data_filtered)
nrow(data_filtered)

data2 <- data_filtered %>%
  mutate(
    # parse time column (ex: "1946-03-26T17:09:10.980Z")
    time_parsed = ymd_hms(time),
    # change to Python's format: "YYYY/MM/DD HH:mm:ss000"
    time_str    = format(time_parsed, "%Y/%m/%d %H:%M:%S000")
  ) %>%
  # extract the columns needed
  select(time_str, latitude, longitude)

head(data2)

write.table(
  data2,
  file = "C:/Users/HP/Downloads/sumatra_earthquake.txt",
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)


###Visualisation
library(ggplot2)
library(dplyr)
library(maps)

world_map <- map_data("world")

p <- ggplot() +
  # Coastline
  geom_polygon(data = world_map,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "grey50", linewidth = 0.3) +
  # Earthquake event
  geom_point(data = data_filtered,
             aes(x = longitude, y = latitude),
             color = "black", size = 0.6, alpha = 0.55) +
  coord_fixed(xlim = c(95, 106), ylim = c(-6.2, 6.2), expand = FALSE) +
  scale_x_continuous(breaks = seq(96, 106, 2)) +
  scale_y_continuous(breaks = seq(-6, 6, 2)) +
  labs(x = "long", y = "lat") +
  theme_bw(base_size = 12) +
  theme(
    panel.grid   = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.text    = element_text(color = "black"),
    axis.ticks   = element_line(color = "black"),
    plot.margin  = margin(8, 12, 8, 8)
  )

print(p)

# Save to .eps
ggsave(
  filename = "C:/Users/HP/Downloads/sumatra_earthquakes.eps",
  plot     = p,
  device   = cairo_ps,
  width    = 7,
  height   = 7,
  units    = "in",
  dpi      = 600,
  fallback_resolution = 600
)
