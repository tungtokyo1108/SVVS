library(phyloseq)
library(phytools)
library(ape)
library(dplyr)
library(tidyr)
library(stringi)
library(stringr)
library(ggtree)

tree_full <- read_tree_greengenes("datasetA_tree.nwk")
plot(tree_full, show.tip.label = F, use.edge.length = T,show.node.label=FALSE,
     direction="downwards", cex = 0.8, type = "unroot", label.offset = 0.05)
otu_full <- tree_full$tip.label

###############################################################################

group_C <- read.csv("otu_datasetA_C.csv")
group_D <- read.csv("otu_datasetA_D.csv")

otu_0 <- group_C$X
otu_1 <- group_D$X

edge_full <- tree_full$edge

tree_full_new <- tree_full
otu_sub_0_in_otu_full <- otu_full %in% otu_0
otu_sub_1_in_otu_full <- otu_full %in% otu_1

otu_full_0 <- rep("", length(otu_full))
otu_full_0[otu_sub_0_in_otu_full] <- rep("+", sum(otu_sub_0_in_otu_full))
otu_full_0[otu_sub_1_in_otu_full] <- rep("*", sum(otu_sub_1_in_otu_full))

tree_full_new$tip.label <- otu_full_0

tipcol <- rep("black", length(tree_full_new$tip.label))
for (i in 1:length(tree_full_new$tip.label)) {
  if (tree_full_new$tip.label[i] == "+") {
    # CDI Case
    tipcol[i] <- "red"
  } else if (tree_full_new$tip.label[i] == "*") {
    # Health control
    tipcol[i] <- "blue"
  } 
}

plot(tree_full_new, type =  "fan", label.offset = 0.05,
     show.tip.label = T, cex = 0.9, tip.color = tipcol)
add.scale.bar()

####

edgecol <- rep('black', nrow(tree_full_new$edge))
edgecol[1:10] <- "black"
edgecol[12:943] <- "green"
edgecol[945:1448] <- "orange" 
edgecol[1496:1737] <- "purple"
plot(tree_full_new, edge.color = edgecol, type = "fan", label.offset = 0.05,
     show.tip.label = T, cex = 1, tip.color = tipcol)
add.scale.bar()

###############################################################################

