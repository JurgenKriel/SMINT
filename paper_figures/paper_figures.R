library(stringr)
library(sf)
library(sfheaders)
library(geomander)
library(lwgeom)
library(ggsankey)
library(UpSetR)
library(pheatmap)
library(RColorBrewer)
library(edgeR)
library(plyr)

source('plot_annotation.R')

unified <- readRDS('unified_spe.rds')
nuc_spe <- readRDS('nucleus_spe.rds')
cell_spe <- readRDS('cell_spe.rds')
cell_poly <- readRDS('cell_polygons.rds')
exp_poly <- readRDS('expansion_polygons.rds')
nuc_poly <- readRDS('nucleus_polygons.rds')

df <- make_df(nuc_spe)
df_cell <- make_df(cell_spe)
udf <- make_df(unified)

####Figure 1A####
plot_annotation(df, annotation = df$nucleus_annotation, colours = cols_anno, 
                subset = df$nucleus_annotation == df$cell_annotation, 
                x_coord =df$x_centroid, y_coord = df$y_centroid) +coord_equal()

####Figure 1C####

exp_poly$x_centroid <- unlist(lapply(st_geometry(st_centroid(exp_poly)), function(x) x[1]))
exp_poly$y_centroid <- unlist(lapply(st_geometry(st_centroid(exp_poly)), function(x) x[2]))
  
ggplot(exp_poly[exp_poly$x_centroid<3600 & exp_poly$y_centroid<3450 & 
                  exp_poly$x_centroid>3420 & exp_poly$y_centroid>3150,], aes(fill = annotation))+
  scale_fill_manual(values = cols_anno)+
  geom_sf()+theme_classic()

ggplot(nuc_poly[nuc_poly$x_centroid<3600 & nuc_poly$y_centroid<3450 & 
                  nuc_poly$x_centroid>3420 & nuc_poly$y_centroid>3150,], aes(fill = nucleus_annotation))+
  scale_fill_manual(values = cols_anno)+
  geom_sf(data = cell_poly[cell_poly$x_centroid<3600 & cell_poly$y_centroid<3450 & 
                      cell_poly$x_centroid>3420 & cell_poly$y_centroid>3150,], aes(fill = cell_annotation),
          alpha = 0.6)+
  geom_sf()+theme_classic()

####Figure 1D####

#find maximum inscribed circle - process in groups of 1024 as st_circle_center corrupts above larger values 
cell_poly$group <- plyr::round_any(1:nrow(cell_poly), 1024, f = ceiling)/1024
cell_centre <- do.call(rbind, lapply(split(cell_poly, cell_poly$group), function(x) st_circle_center(x)))
cell_inscribe <- st_buffer(cell_centre, st_distance(cell_centre, st_cast(cell_poly, 'MULTILINESTRING'), by_element = T))

exp_poly$area <- st_area(exp_poly)
exp_poly$group <- plyr::round_any(1:nrow(exp_poly), 1024, f = ceiling)/1024
exp_centre <- do.call(rbind, lapply(split(exp_poly, exp_poly$group), function(x) st_circle_center(x)))
exp_inscribe <- st_buffer(exp_centre, st_distance(exp_centre, st_cast(exp_poly, 'MULTILINESTRING'), by_element = T))

#find minimum bounding circle
cell_bound <- st_minimum_bounding_circle(cell_poly)
exp_bound <- st_minimum_bounding_circle(exp_poly)

#calculate ratio of bounding and inscribed circle perimeter
cell_poly$cell_ratio <- st_perimeter_lwgeom(cell_inscribe)/st_perimeter_lwgeom(cell_bound)
exp_poly$exp_ratio <- st_perimeter_lwgeom(exp_inscribe)/st_perimeter_lwgeom(exp_bound)

cell_poly <- st_make_valid(cell_poly)
exp_poly <- st_make_valid(exp_poly)
#find matching cells from alternate segmentations
sf_int <- st_intersection(exp_poly, cell_poly)
sf_int$prop <- st_area(sf_int)/sf_int$area.1
sf_int <- sf_int[sf_int$prop >=0.45,]

#find number of cells within 15 µm of cell : average between both segmentation strategies
cellb <- st_buffer(st_centroid(cell_poly), 15)
stj <- st_join(cellb, cell_poly, left = F)
cell_poly$nn_15 <- as.numeric(table(stj$polygon_id.x))
expb <- st_buffer(st_centroid(exp_poly), 15)
stj <- st_join(expb, exp_poly, left = F)
exp_poly$nn_15 <- as.numeric(table(stj$cell_id.x))
sf_int$xen_nn_15 <- exp_poly$nn_15[match((sf_int$cell_id),exp_poly$cell_id)]
sf_int$cell_nn_15 <- cell_poly$nn_15[match((sf_int$polygon_id),cell_poly$polygon_id)]
sf_int$ave_nn <- (sf_int$cell_nn_15+sf_int$xen_nn_15)/2

ggplot(sf_int, aes(area.1, area,col = ave_nn-1))+geom_point(size =0.1)+
  theme_minimal()+scale_color_viridis_c(option = 'magma', values = c(0,0.05,0.1,0.2,0.3, 0.4,1))+
  geom_abline(col = 'red')+scale_y_log10()+scale_x_log10()

####Figure 1E####
ggplot(sf_int, aes(cell_ratio, exp_ratio, col = ave_nn-1))+geom_point(size =0.1)+
  theme_minimal()+scale_color_viridis_c(option = 'magma', values = c(0,0.05,0.1,0.2,0.3, 0.4,1))+
  geom_abline(col = 'red')

####Figure 2A####
#Figure 2A i
#remove smaller populations as changes cannot be seen relative to larger populations
dfl <- df[!df$cell_annotation %in% c('Mural', 'NPC-like', 'Macrophage', 'OPC', 'Lymphoid', 'Fibroblast') &
            !df$nucleus_annotation %in% c('Mural', 'NPC-like', 'Macrophage', 'OPC', 'Lymphoid', 'Fibroblast'),] %>% 
  make_long(nucleus_annotation, cell_annotation)

ggplot(dfl, aes(x = x,
                next_x = next_x,
                node = node,
                next_node = next_node,
                fill = factor(node))) +
  geom_sankey(alpha = 0.8)+theme_void()+scale_fill_manual(values = cols_anno)

#supplementary: smaller populations 
dfl <- df[df$cell_annotation %in% c('Mural', 'NPC-like', 'Macrophage', 'OPC', 'Lymphoid', 'Fibroblast') |
            df$nucleus_annotation %in% c('Mural', 'NPC-like', 'Macrophage', 'OPC', 'Lymphoid', 'Fibroblast'),] %>% 
  make_long(nucleus_annotation, cell_annotation)

ggplot(dfl, aes(x = x,
                next_x = next_x,
                node = node,
                next_node = next_node,
                fill = factor(node))) +
  geom_sankey(alpha = 0.8)+theme_void()+scale_fill_manual(values = cols_anno)

#Figure 2A ii
plot_annotation(df, annotation = df$nucleus_annotation, colours = cols_anno)+coord_equal()

#Figure 2A iii
plot_annotation(df_cell, annotation = df_cell$cell_annotation, colours = cols_anno)+coord_equal()

#Figure 2A iv
plot_annotation(df, annotation = df$nucleus_annotation, colours = cols_anno, 
                subset = df$nucleus_annotation != df$cell_annotation, 
                x_coord =df$x_centroid, y_coord = df$y_centroid) +coord_equal()

#Figure 2A v
plot_annotation(df, annotation = df$cell_annotation, colours = cols_anno, 
                subset = df$nucleus_annotation != df$cell_annotation, 
                x_coord =df$x_centroid, y_coord = df$y_centroid) +coord_equal()

####Figure 2B####
plot_annotation(df, annotation = NULL,subset = df$nucleus_annotation != df$cell_annotation, 
                x_coord =df$x_centroid, y_coord = df$y_centroid) +coord_equal()+
  geom_density_2d_filled(data = df, aes(x_centroid, y_centroid),h =c(400,400), alpha =0.95)+
  xlim(0,10000)+ylim(0,9000)+theme_classic()

####Figure 2C####
plot_annotation(df, annotation = NULL,subset = df$nucleus_annotation != df$cell_annotation, 
                x_coord =df$x_centroid, y_coord = df$y_centroid) +coord_equal()+
  geom_density_2d_filled(data = df[df$nucleus_annotation != df$cell_annotation,],
                         aes(x_centroid, y_centroid),h =c(400,400), alpha =0.95)+
  xlim(0,10000)+ylim(0,9000)+theme_classic()

####Figure 2D####

####WARNING: The pairwise comparisons of this step take a long time to run. #####
tab_df <- list()
for(i in rownames(nuc_spe)){
  for(j in rownames(nuc_spe)){
    tab_df[paste0(i, '_', j)] <- sum(colSums(counts(nuc_spe)[c(i,j),]>0)==2)/sum(colSums(counts(nuc_spe)[c(i,j),]>0)>0)}
}
tab_df <- as.data.frame(unlist(tab_df))


cell_df <- list()
for(i in rownames(nuc_spe)){
  for(j in rownames(nuc_spe)){
    cell_df[paste0(i, '_', j)] <- sum(colSums(counts(cell_spe)[c(i,j),]>0)==2)/sum(colSums(counts(cell_spe)[c(i,j),]>0)>0)}
}
tab_df <- cbind(tab_df, as.data.frame(unlist(cell_df)))
rownames(tab_df) <- c('nuc', 'cell')

ggplot(tab_df, aes(nuc, cell))+geom_point()+theme_minimal()
      #repeat with single cell and expansion data, and subset to only gene pairs with <1% co-expression in single cell


####Figure 2E####
sc <- readRDS('path/to/sc_reference.rds')
expe <-  readRDS('path/to/expansion_data.rds')

nuc_A <- table(colSums(counts(nuc_spe['ARHGDIB',])>0), 
              nuc_spe$nucleus_annotation)[2,]/colSums(table(colSums(counts(nuc_spe['ARHGDIB',])>0), 
                                                            nuc_spe$nucleus_annotation))
nuc_M <- table(colSums(counts(nuc_spe['MOBP',])>0), 
              nuc_spe$nucleus_annotation)[2,]/colSums(table(colSums(counts(nuc_spe['MOBP',])>0), 
                                                            nuc_spe$nucleus_annotation))
cell_A <- table(colSums(counts(cell_spe['ARHGDIB',])>0), 
                cell_spe$cell_annotation)[2,]/colSums(table(colSums(counts(cell_spe['ARHGDIB',])>0),
                                                            cell_spe$cell_annotation))
cell_M <- table(colSums(counts(cell_spe['MOBP',])>0), 
                cell_spe$cell_annotation)[2,]/colSums(table(colSums(counts(cell_spe['MOBP',])>0),
                                                            cell_spe$cell_annotation))

sc_A <- table(colSums(counts(sc['ARHGDIB',])>0), sc$annotation)[2,]/colSums(table(colSums(counts(sc['ARHGDIB',])>0), sc$annotation))
sc_M <- table(colSums(counts(sc['MOBP',])>0), sc$annotation)[2,]/colSums(table(colSums(counts(sc['MOBP',])>0), sc$annotation))
expe_A <- table(colSums(counts(expe['ARHGDIB',])>0), expe$annotation)[2,]/colSums(table(colSums(counts(expe['ARHGDIB',])>0), expe$annotation))
expe_M <- table(colSums(counts(expe['MOBP',])>0), expe$annotation)[2,]/colSums(table(colSums(counts(expe['MOBP',])>0), expe$annotation))

df_sp <- data.frame(nuc_A=as.numeric(nuc_A), nuc_M=as.numeric(nuc_M), cell_A=as.numeric(cell_A), cell_M=as.numeric(cell_M), celltype = names(cell_M))

ggplot(df_sp, aes(celltype, -nuc_A, fill =celltype))+theme_minimal()+
  scale_fill_manual(values = cols_anno)+scale_color_manual(values = cols_anno)+
  #geom_bar(aes(y = sc_M, col = celltype), stat = 'identity', alpha = 0)+
  #geom_bar(aes(y = -sc_A, col= celltype), stat = 'identity', alpha = 0)+
  geom_bar(aes(y = cell_M), stat = 'identity', alpha = 0.6)+
  geom_bar(aes(y = -cell_A), stat = 'identity', alpha = 0.6)+
  geom_bar(aes(y = nuc_M), stat = 'identity')+
  geom_bar(stat= 'identity')+
  #geom_bar(aes(y = sc_M), stat = 'identity', alpha = 0, col='black')+
  #geom_bar(aes(y = -sc_A), stat = 'identity', alpha = 0, col ='black')+
  #geom_bar(data = df_sp[df_sp$celltype == 'Endothelial',], 
  #   aes(y = -sc_A), stat = 'identity', alpha = 0, col ='white')+
  #geom_bar(data = df_sp[df_sp$celltype == 'Endothelial',], 
  #   aes(y = sc_M), stat = 'identity', alpha = 0, col ='white')+
  geom_hline(yintercept = 0)


####Figure 3A####
df_cell$n_nuclei <- table(nuc_spe$cellmatch)[as.character(colnames(cell_spe))]
df_cell$log_nuc <- log10(table(df_cell$n_nuclei)+0.1)[as.character(df_cell$n_nuclei)]/ table(df_cell$n_nuclei)[as.character(df_cell$n_nuclei)]
ggplot(df_cell, aes(x = n_nuclei, y = log_nuc, fill = cell_annotation))+ geom_bar( stat = 'identity')+scale_fill_manual(values = cols_anno)+theme_void()

####Figure 3B####
#all cells with >1 nucleus
res <- table(cell_spe$cell_annotation, colnames(cell_spe) %in% nuc_spe$cellmatch[duplicated(nuc_spe$cellmatch)])
res_enrich <- res/chisq.test(res)$expected

#all cells with >1 nucleus and annotations match between them all
res2 <- table(cell_spe$cell_annotation,
              colnames(cell_spe) %in% nuc_spe$cellmatch[duplicated(nuc_spe$cellmatch)] &
                rowSums(table(nuc_spe$cellmatch, nuc_spe$nucleus_annotation)>0)[colnames(cell_spe)]==1 &
                sort(unique(cell_spe$cell_annotation))[apply(table(nuc_spe$cellmatch, 
                                                                   nuc_spe$nucleus_annotation)[colnames(cell_spe),], 
                                                             MARGIN=1, which.max)] == cell_spe$cell_annotation)
res2_enrich <- res2/chisq.test(res2)$expected

#all cells with 3 or more nuclei
res3 <- table(cell_spe$cell_annotation, colnames(cell_spe) %in% names(which(table(nuc_spe$cellmatch)>=3)))
res3_enrich <- res3/chisq.test(res3)$expected

df_multi <- data.frame(rowSums(res),res_enrich[,2],res2_enrich[,2],res3_enrich[,2])
colnames(df_multi) <- c('observed', 'all_multi_prop', 'true_multi_prop', 'multi3_prop')
df_multi$celltype <- factor(rownames(df_multi), levels = rownames(df_multi)[order(df_multi$observed)])
df_multi$plus3_multi_p <- pchisq(((res3-chisq.test(res3)$expected)^2/chisq.test(res3)$expected)[,2], 1, lower.tail =F)
df_multi$all_multi_p <- pchisq(((res-chisq.test(res)$expected)^2/chisq.test(res)$expected)[,2], 1, lower.tail =F)
df_multi$cons_multi_p <- pchisq(((res2-chisq.test(res2)$expected)^2/chisq.test(res2)$expected)[,2], 1, lower.tail =F)
df_multi$all_multi_sig[df_multi$all_multi_p<0.05] <- '*'
df_multi$all_multi_sig[df_multi$all_multi_p<0.01] <- '**'
df_multi$all_multi_sig[df_multi$all_multi_p<0.001] <- '***'
df_multi$all_multi_sig[is.na(df_multi$all_multi_sig)] <- ''
df_multi$plus3_multi_sig[df_multi$plus3_multi_p<0.05] <- '*'
df_multi$plus3_multi_sig[df_multi$plus3_multi_p<0.01] <- '**'
df_multi$plus3_multi_sig[df_multi$plus3_multi_p<0.001] <- '***'
df_multi$plus3_multi_sig[is.na(df_multi$plus3_multi_sig)] <- ''
df_multi$cons_multi_sig[df_multi$cons_multi_p<0.05] <- '*'
df_multi$cons_multi_sig[df_multi$cons_multi_p<0.01] <- '**'
df_multi$cons_multi_sig[df_multi$cons_multi_p<0.001] <- '***'
df_multi$cons_multi_sig[is.na(df_multi$cons_multi_sig)] <- ''

ggplot(df_multi, aes(celltype, true_multi_prop, col = celltype, group = celltype))+
  geom_segment(aes(y =rowMaxs(as.matrix(df_multi[,c(2:4)])), yend = rowMins(as.matrix(df_multi[,c(2:4)]))))+
  scale_color_manual(values = cols_anno)+theme_minimal()+scale_y_log10()+geom_hline(yintercept = 1)+
  geom_point(aes(y =multi3_prop), size =4.3, shape =1, stroke =1)+
  geom_point(aes(y =multi3_prop), size =3.7, col = 'white')+geom_point(aes(y =all_multi_prop), size =4.3)+
  geom_point( size =4.3, col ='black')+geom_point( size =2.9, col ='white')+geom_point(size =2.3)+theme_classic()+
  geom_text( aes(label =cons_multi_sig, y = 0.965*true_multi_prop), col = 'black', nudge_x = 0.25, hjust = 0, size = 5)+
  theme_classic()+geom_text( aes(label =plus3_multi_sig, y = 0.965*multi3_prop), nudge_x = 0.25, hjust = 0, size = 5, fontface = 'italic')+
  geom_text( aes(label =all_multi_sig, y = 0.965*all_multi_prop), nudge_x = -0.25, hjust = 1, size = 5)

####Figure 3C####

    #Inconsistent nuclei annotation
ggplot(df_cell[df_cell$n_nuclei>1 & table(nuc_spe$nucleus_annotation ==nuc_spe$cell_annotation, 
                                          nuc_spe$cellmatch)[1,as.character(colnames(cell_spe))]>0,], 
       aes(x = x_centroid, y = y_centroid))+ geom_density_2d_filled(h =c(1200,1200), alpha =0.95)+
  xlim(0,10000)+ylim(0,9000)+theme_classic()+theme_void()+coord_equal()

    #Consistent nuclei annotation
ggplot(df_cell[df_cell$n_nuclei>1 & table(nuc_spe$nucleus_annotation ==nuc_spe$cell_annotation, 
                                          nuc_spe$cellmatch)[1,as.character(colnames(cell_spe))]==0,], 
       aes(x = x_centroid, y = y_centroid))+ geom_density_2d_filled(h =c(1000,1000), alpha =0.95)+
  xlim(0,10000)+ylim(0,9000)+theme_classic()+theme_void()+coord_equal()

####Figure 3D####
multi <- as.data.frame.array(table(nuc_spe$cellmatch[nuc_spe$cellmatch %in% nuc_spe$cellmatch[which(duplicated(nuc_spe$cellmatch))]],
                                   nuc_spe$nucleus_annotation[nuc_spe$cellmatch %in% nuc_spe$cellmatch[which(duplicated(nuc_spe$cellmatch))]]))
upset(as.data.frame.array(1* (multi>0)), nsets = 16, order.by='freq', nintersects = 33)

####Figure 3D i####
ggplot(nuc_poly[nuc_poly$x_centroid<8750 & nuc_poly$y_centroid<5150 & 
                  nuc_poly$x_centroid>8350 & nuc_poly$y_centroid>4700 &
                  nuc_poly$cellmatch %in% cell_poly$polygon_id[cell_poly$x_centroid<8750 & 
                                                                 cell_poly$y_centroid<5150 & 
                                                                 cell_poly$x_centroid>8350 & 
                                                                 cell_poly$y_centroid>4700 & 
                                                                 df_cell$n_nuclei>1 &
                                                                 table(nuc_spe$nucleus_annotation ==nuc_spe$cell_annotation, 
                                                                         nuc_spe$cellmatch)[1,as.character(colnames(cell_spe))]==0],], 
       aes(fill = nucleus_annotation))+
  scale_fill_manual(values = cols_anno)+
  geom_sf(data = cell_poly[cell_poly$x_centroid<8750 & cell_poly$y_centroid<5100 & 
                             cell_poly$x_centroid>8350 & cell_poly$y_centroid>4700,], fill = 'white')+
  geom_sf(data = cell_poly[cell_poly$x_centroid<8750 & cell_poly$y_centroid<5150 & 
                             cell_poly$x_centroid>8350 & cell_poly$y_centroid>4700 & 
                             df_cell$n_nuclei>1 & table(nuc_spe$nucleus_annotation ==nuc_spe$cell_annotation, 
                                                        nuc_spe$cellmatch)[1,as.character(colnames(cell_spe))]==0,], 
          aes(fill = cell_annotation),
          alpha = 0.6)+
  geom_sf()+theme_minimal()

####Figure 3D ii####
ggplot(nuc_poly[nuc_poly$x_centroid<5200 & nuc_poly$y_centroid<3600 & 
                  nuc_poly$x_centroid>4800 & nuc_poly$y_centroid>3200 &
                  nuc_poly$cellmatch %in% cell_poly$polygon_id[cell_poly$x_centroid<5200 & 
                                                                 cell_poly$y_centroid<3600 & 
                                                                 cell_poly$x_centroid>4800 & 
                                                                 cell_poly$y_centroid>3200 & 
                                                                 df_cell$n_nuclei>1 &
                                                                 table(nuc_spe$nucleus_annotation ==nuc_spe$cell_annotation, 
                                                                       nuc_spe$cellmatch)[1,as.character(colnames(cell_spe))]==0],], 
       aes(fill = nucleus_annotation))+
  scale_fill_manual(values = cols_anno)+
  geom_sf(data = cell_poly[cell_poly$x_centroid<5200 & cell_poly$y_centroid<3600 & 
                             cell_poly$x_centroid>4800 & cell_poly$y_centroid>3200,], fill = 'white')+
  geom_sf(data = cell_poly[cell_poly$x_centroid<5200 & cell_poly$y_centroid<3600 & 
                             cell_poly$x_centroid>4800 & cell_poly$y_centroid>3200 & 
                             df_cell$n_nuclei>1 & table(nuc_spe$nucleus_annotation ==nuc_spe$cell_annotation, 
                                                        nuc_spe$cellmatch)[1,as.character(colnames(cell_spe))]==0,], 
          aes(fill = cell_annotation),
          alpha = 0.6)+
  geom_sf()+theme_minimal()

####Figure 4B####
#Figure 4B i
plot_annotation(df, annotation = df$nucleus_annotation, colours = cols_anno, 
                x_coord =df$x_centroid, y_coord = df$y_centroid) +coord_equal()

#Figure 4B ii
nuc_spe$ybin <- round_any(nuc_spe$y_centroid, 40, f = floor)
nuc_spe$xbin <- round_any(nuc_spe$x_centroid, 40, f = floor)
bins <- apply(table(paste0(nuc_spe$xbin, '_', nuc_spe$ybin), nuc_spe$neighborhood),MARGIN = 1, which.max)
dfb <- data.frame(neighborhood = bins, x_grid = as.numeric(gsub('_.*', '', names(bins))), y_grid = as.numeric(gsub('.*_', '', names(bins))))


cols <- brewer.set1(9)
ggplot(dfb, aes(x_grid, y_grid, fill = as.factor(neighborhood)))+
  geom_tile(height =40, width =40)+coord_equal()+theme_classic()+scale_fill_manual(values = cols)


#Figure 4B iii
make_log_map(unified, 'X234.01448059082', udf, order =T)+scale_color_viridis_c(option = 'magma', values = c(0,0.4,0.5, 0.7,1))+coord_equal()

#Figure 4B iv
plot_annotation(udf, annotation = udf$Cluster,
                x_coord =udf$x_centroid, y_coord = udf$y_centroid) +coord_equal()


####Figure 4C####
res_n <- table(nuc_spe$neighborhood, nuc_spe$nucleus_annotation)
rownames(res_n) <- paste0('N',rownames(res_n))
res_n_enrich <- res_n/chisq.test(res_n)$expected

unified$Anno[str_detect(unified$Anno, 'Cell')] <- 'Lymphoid'
unified$Anno[str_detect(unified$Anno, 'Microglia')] <- 'Microglia'
unified$Anno[str_detect(unified$Anno, 'AC-like')] <- 'AC-like'

res_u <- table(unified$Cluster, unified$Anno)
rownames(res_u) <- paste0('U',rownames(res_u))
res_u_enrich <- res_u/chisq.test(res_u)$expected

res_heatmap <- rbind(res_n_enrich, res_u_enrich[1:12,colnames(res_n)])

pheatmap(res_heatmap, color = c(rev(colorRampPalette(brewer.pal(9, "Blues"))(10)),
                                colorRampPalette(brewer.pal(9, "YlOrRd"))(9)),
         breaks=c(seq(0, 1, 0.1), seq(2, 5, 1), seq(6, 10, 1)))

####Figure 4D####
plot_annotation(udf, annotation = udf$Anno, colours = cols_anno) +coord_equal()

####Figure 4E####
plot_annotation(udf, annotation = udf$Cluster) +coord_equal()

####Figure 5B####
plot_annotation(df, annotation = df$nucleus_annotation, colours = cols_anno, 
                x_coord =df$x_centroid, y_coord = df$y_centroid,
                celltypes = 'OPC-like', outline = T) +coord_equal()

####Figure 5C####
d0 <- DGEList(counts(nuc_spe[,!is.na(nuc_spe$neighborhood)]))
d0 <- calcNormFactors(d0)
group <- interaction(nuc_spe$neighborhood[!is.na(nuc_spe$neighborhood)], gsub('-', '_',nuc_spe$nucleus_annotation[!is.na(nuc_spe$neighborhood)]))
mm <- model.matrix(~0 + group)
tmp <- voom(d0, mm, plot=F)
fit <- lmFit(tmp, mm)
contr <- makeContrasts((group7.OPC_like+group6.OPC_like)/2 - (group8.OPC_like+group3.OPC_like+group9.OPC_like)/3, levels = colnames(coef(fit)))
tmp <- contrasts.fit(fit, contr)
tmp <- eBayes(tmp)
top.table <- topTable(tmp, sort.by = "P", n = Inf)

ggplot(top.table, aes(logFC, -log10(adj.P.Val)))+geom_point()+geom_hline(yintercept = -log10(0.05), linetype =2, col = 'grey')+theme_classic()+
  geom_point(data =top.table[rownames(top.table) %in% c('EGFR', 'PDGFRA') ,], col = 'black', size =2.5)+
  geom_point(data =top.table[top.table$adj.P.Val<0.05 & top.table$logFC >0.2 ,], col = 'blue')+
  geom_point(data =top.table[top.table$adj.P.Val<0.05 & top.table$logFC < -0.2 ,], col = 'red')+
  geom_point(data =top.table[rownames(top.table) %in% c('EGFR', 'PDGFRA') ,], col = 'yellow', size =1.8)

####Figure 5D####
make_log_map(nuc_spe, 'EGFR', df, order =T)+coord_equal()

####Figure 5E####
make_log_map(unified[,counts(unified)['X296.066253662109',] <1500], 'X296.066253662109', 
             udf[counts(unified)['X296.066253662109',] <1500,], order =T)+
  scale_color_viridis_c(option = 'magma',limits = c(0, 1500), values = c(0,0.4,0.5, 0.7,1))+coord_equal()






sessionInfo()