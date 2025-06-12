# import ggplot2
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
# import data from arguments
data_base <- read.table(args[1], header = TRUE, sep = ",")

data_mine <- read.table(args[2], header = TRUE, sep = ",")

# plot precision, recall, f1, real_precision, real_recall, real_f1
# Their names are Precision@0.5, Recall@0.5, F1@0.5, Precision, Recall, F1
# The x-axis is from 0.3 to 0.9
# The y-axis is from 0 to 1
# The legend of YOLOv8 is blue and the legend of DDPG is red. The legend of DDPG is on the right side of the plot.
# Bigger font size, thicker line
# Plot each of them in a different plot and save them as png files

x_range <- seq(0.3, 0.9, 0.1)
# Plot precision
p <- ggplot(data_base, aes(x = x_range, y = precision), color = "blue") +
  geom_line(linewidth = 2) +
  geom_line(data = data_mine, aes(x = x_range, y = precision), color = "red", linewidth = 2) +
  theme_minimal() +
  labs(x = "Confidence Threshold", y = "Precision@0.5") + 
  theme(legend.position = "right") +
  theme(text = element_text(size = 80)) + 
  ylim(0, 1)
ggsave(p, file = "figures/precision.png", dpi = 300, width = 20, height = 10)

# Plot recall
p <- ggplot(data_base, aes(x = x_range, y = recall), color = "blue") +
  geom_line(linewidth = 2) +
  geom_line(data = data_mine, aes(x = x_range, y = recall), color = "red", linewidth = 2) +
  theme_minimal() +
  labs(x = "Confidence Threshold", y = "Recall@0.5") +
  theme(legend.position = "right") +
  theme(text = element_text(size = 80)) + 
  ylim(0, 1)
ggsave(p, file = "figures/recall.png", dpi = 300, width = 20, height = 10)

# Plot f1
p <- ggplot(data_base, aes(x = x_range, y = f1), color = "blue") +
  geom_line(linewidth = 2) +
  geom_line(data = data_mine, aes(x = x_range, y = f1), color = "red", linewidth = 2) +
  theme_minimal() +
  labs(x = "Confidence Threshold", y = "F1@0.5") +
  theme(legend.position = "right") +
  theme(text = element_text(size = 80)) + 
  ylim(0, 1)
ggsave(p, file = "figures/f1.png", dpi = 300, width = 20, height = 10)

# Plot real_precision
p <- ggplot(data_base, aes(x = x_range, y = real_precision), color = "blue") +
  geom_line(linewidth = 2) +
  geom_line(data = data_mine, aes(x = x_range, y = real_precision), color = "red") +
  theme_minimal() +
  labs(x = "Confidence Threshold", y = "Precision") +
  theme(legend.position = "right") +
  theme(text = element_text(size = 80)) + 
  ylim(0, 1)
ggsave(p, file = "figures/real_precision.png", dpi = 300, width = 20, height = 10)

# Plot real_recall
p <- ggplot(data_base, aes(x = x_range, y = real_recall), color = "blue") +
  geom_line(linewidth = 2) +
  geom_line(data = data_mine, aes(x = x_range, y = real_recall), color = "red") +
  theme_minimal() +
  labs(x = "Confidence Threshold", y = "Recall") +
  theme(legend.position = "right") +
  theme(text = element_text(size = 80)) + 
  ylim(0, 1)
ggsave(p, file = "figures/real_recall.png", dpi = 300, width = 20, height = 10)

# Plot real_f1
p <- ggplot(data_base, aes(x = x_range, y = real_f1), color = "blue") +
  geom_line(linewidth = 2) +
  geom_line(data = data_mine, aes(x = x_range, y = real_f1), color = "red") +
  theme_minimal() +
  labs(x = "Confidence Threshold", y = "F1") +
  theme(legend.position = "right") +
  theme(text = element_text(size = 80)) + 
  ylim(0, 1)
ggsave(p, file = "figures/real_f1.png", dpi = 300, width = 20, height = 10)