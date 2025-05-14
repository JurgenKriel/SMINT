library(sf)
library(sfheaders)
library(stringr)
library(Seurat)
library(SpatialExperiment)
library(ggplot2)
library(forcats)
library(dplyr)
library(data.table)

#load aligned spatial metabolomic and spatial transcriptomic outputs

SM <- as.data.frame(fread('/path/to/aligned/metabcoordinates.csv', sep = ','))
spe <- readRDS('path/to/ST_object.rds')

#analysis column names of SM and determine number of metabolites
colnames(SM)
which(str_detect(colnames(SM), 'X'))
num_metabs <- 451

#load and create spatial features objects of both modalities

ST_sf <- readRDS('path/to/ST_polygons.rds')
SM_sf <- sf_point(ven2c_out, x = 'x_new', y = 'y_new', keep =T)

#join all metabolite gridpoints within 20 µm of polygon centroid to that polygon
joint <- st_join(st_buffer(st_centroid(ST_sf), 20), SM_sf, left = F)
joint <- as.data.frame(joint)

#collect average normalised abundance of each metabolite across all gridpoints within 20µm of each polygon
sparse_mat <- as.sparse(joint[, which(str_detect(colnames(joint), 'X'))])
sparse_mat <- as.sparse(matrix(unlist(lapply(lapply(split(sparse_mat, joint$cell_id), matrix, ncol = num_metabs), colMeans)), nrow = num_metabs))

#convert matrix into SpatialExperiment object
rownames(sparse_mat) <- colnames(joint)[which(str_detect(colnames(joint), 'X'))]
colnames(sparse_mat) <- unique(joint$cell_id)
smpe <- SpatialExperiment(assay = list('counts' = sparse_mat))

#combine ST and SM results into single spatial experiment object
#As SM data has already been normalized, we combine SM with the ST lognormalized counts
combined_counts <- rbind(counts(smpe), logcounts(spe[,colnames(smpe)]))

smpe <- SpatialExperiment(assay = list('logcounts' = combined_counts))
colData(smpe) <- cbind(colData(smpe), colData(spe)[colnames(smpe),])

saveRDS(smpe, 'path/to/save_location.rds')


