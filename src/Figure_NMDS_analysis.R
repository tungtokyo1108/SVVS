library(phyloseq)
library(ggplot2)

res <- read.csv("NMDS_datasetA.csv")
res$EM_labels <- as.factor(res$EM_labels)
res$SVVS_Labels <- as.factor(res$SVVS_Labels)
res$True_labels <- as.factor(res$True_labels)

p <- ggplot(res, aes(MDS1, MDS2, colour = EM_labels)) + geom_point(size=3) + theme_bw() +
  theme(plot.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) + 
  theme(axis.line = element_line(color = "black")) + 
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15),
    axis.title.y = element_text(size = 16)) +
  scale_color_manual(values = c("red", "blue"))
p + labs(x = "NMDS1", y = "NMDS2")

p <- ggplot(res, aes(MDS1, MDS2, colour = SVVS_Labels)) + geom_point(size=3) + theme_bw() +
  theme(plot.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) + 
  theme(axis.line = element_line(color = "black")) + 
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15),
    axis.title.y = element_text(size = 16)) +
  scale_color_manual(values = c("red", "blue"))
p + labs(x = "NMDS1", y = "NMDS2")

p <- ggplot(res, aes(MDS1, MDS2, colour = True_labels)) + geom_point(size=3) + theme_bw() +
  theme(plot.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) + 
  theme(axis.line = element_line(color = "black")) + 
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15),
    axis.title.y = element_text(size = 16)) +
  scale_color_manual(values = c("red", "blue"))
p + labs(x = "NMDS1", y = "NMDS2")

######################################################################

res <- read.csv("NMDS_datasetE.csv")
res$SVVS_Labels <- as.factor(res$SVVS_Labels)
res$EM_Labels <- as.factor(res$EM_Labels)

p <- ggplot(res, aes(MDS1, MDS2, colour = SVVS_Labels)) + geom_point(size=3) + theme_bw() +
  theme(plot.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) + 
  theme(axis.line = element_line(color = "black")) + 
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15),
    axis.title.y = element_text(size = 16)) +
  scale_color_manual(values = c("red", "blue"))
p + labs(x = "NMDS1", y = "NMDS2")

p <- ggplot(res, aes(MDS1, MDS2, colour = EM_Labels)) + geom_point(size=3) + theme_bw() +
  theme(plot.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) + 
  theme(axis.line = element_line(color = "black")) + 
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15),
    axis.title.y = element_text(size = 16)) +
  scale_color_manual(values = c("red", "blue"))
p + labs(x = "NMDS1", y = "NMDS2")



















