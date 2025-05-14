
library(sf)
library(sfheaders)
library(stringr)
library(Seurat)
library(SpatialExperiment)
library(ggplot2)
library(forcats)
library(dplyr)
library(data.table)

#load transcripts and aligned polygons

long_out <- fread('/path/to/transcripts.csv.gz')
sf_final <- readRDS('/path/to/aligned_sf.rds')

#ensure polygons are valid

sf_final <- st_make_valid(sf_final)

#filter out poor quality transcripts

long_out <- long_out[long_out$qv>=20,]

long_out_points <- sf_point(long_out, x = 'x_location', y = 'y_location', keep = TRUE)

#join transcripts to polygons

point_out <- st_join(long_out_points, sf_final, left = F)

df_cell <- data.frame(list(id = point_out$cell, gene = point_out$feature_name))

#group transcripts by cell id and create Spatial Experiment Object 

cell_list <- split(df_cell$gene, df_cell$id)
cell_list <- lapply(lapply(cell_list,factor,levels = unique(df_cell$gene)),table)
cell_list <- lapply(cell_list, as.integer)
cell_matrix <- matrix(unlist(cell_list), ncol = length(cell_list), byrow = F)
colnames(cell_matrix) <- names(cell_list)
rownames(cell_matrix) <- unique(df_cell$gene)
sparse_mat <- as.sparse(cell_matrix)
spe_trial <- SpatialExperiment(assay = list('counts' = sparse_mat))

saveRDS(spe_trial, '/path/to/save_spe.rds')

