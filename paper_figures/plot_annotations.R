library(ggplot2)
library(colorspace)
library(pals)
library(viridis)

cols_anno <- c("AC-like"="#F28482", "Endothelial" = "#000000", "Astrocyte" = "#F9C8C7",
               "NPC-like"="#3A7CA5", "Oligodendrocyte" = "#A7C957", "OPC-like"= "#6A994E", "Progenitor"="#EF233C",
               "Macrophage"="#A68A64","Microglia"="#7F4F24", "Mural"="#ADB5BD",
               "MES-like"="#9D4EDD", "OPC"="#8FBFAA","Excitatory" = "#10F4B9",
               "Inhibitory" = "#9CFDFF", 'Fibroblast' = '#B068AA', 'Lymphoid' = '#E85D04')

make_df <- function(spe, reducedDim = 'UMAP', extra_coldata = NULL){
  df <- data.frame(UMAP1=reducedDim(spe, reducedDim)[, 1], UMAP2=reducedDim(spe, reducedDim)[, 2])
  df <- as.data.frame(cbind(df, colData(spe)))
  if(!is.null(extra_coldata)){df <- as.data.frame(cbind(df, extra_coldata))}
  return(df)
}

plot_annotation <- function(df, colours = NULL, annotation, celltypes = NULL, 
                            outline = F, x_coord = NULL, y_coord = NULL, sample_ids = NULL,
                            subset = NULL){
  
  if(is.null(colours)){
      if(length(unique(annotation))<=36){colours <- as.vector(polychrome(length(unique(annotation))))
          }else{
        colours <- sample(colors(), length(unique(annotation)))}
            names(colours) <- unique(annotation)                    
      }
  
  themes <- theme_void()
  if(length(colours)>36){themes <- themes + theme(legend.position = "none")}
  
  
  if(is.null(x_coord)){x_coord <- df$UMAP1}
  if(is.null(y_coord)){y_coord <- df$UMAP2}
  
  if(!is.null(subset)){
    if(is.null(x_coord)){x_coord <- df$UMAP1[subset]}else{x_coord <- x_coord[subset]}
    if(is.null(y_coord)){y_coord <- df$UMAP2[subset]}else{y_coord <- y_coord[subset]}
    if(!is.null(annotation)){annotation <- annotation[subset]}
    df <- df[subset,]
  }
  
  if(!is.null(sample_ids)){
    if(is.null(x_coord)){x_coord <- df$UMAP1[df$id_sample %in% sample_ids]}else{x_coord <- x_coord[df$id_sample %in% sample_ids]}
    if(is.null(y_coord)){y_coord <- df$UMAP2[df$id_sample %in% sample_ids]}else{y_coord <- y_coord[df$id_sample %in% sample_ids]}
    if(!is.null(annotation)){annotation <- annotation[df$id_sample %in% sample_ids]}
    df <- df[df$id_sample %in% sample_ids,]
  }

  samp1 <- sample(1:nrow(df))
  
  
  if(!is.null(celltypes)){ #overlay a specific celltype
    samp2 <- sample(which(annotation %in% celltypes))
    if(outline){ #only include outline of other cell types
      
      out_plot <- ggplot(df[samp1,], aes(x=x_coord[samp1], y=y_coord[samp1])) + 
        geom_point(col="black", size=0.9) +
        geom_point(col="white", size=0.5) +
        geom_point(aes(col=annotation[samp1]), size=0.455, alpha=0.75, col = 'white') +
        geom_point(aes(x=x_coord[samp2], y=y_coord[samp2]),col="black", size=1.1, data = df[samp2,])+
        geom_point(aes(x=x_coord[samp2], y=y_coord[samp2], col =annotation[samp2]), size=0.455, alpha=0.75, data = df[samp2,])+
        scale_color_manual(values=colours) + themes +
        guides(colour = guide_legend(override.aes = list(alpha = 1, size=3)))
      
    }else {     #include colouring of other cell types
      
      out_plot <- ggplot(df[samp1,], aes(x=x_coord[samp1], y=y_coord[samp1])) + 
        geom_point(col="black", size=0.9) +
        geom_point(col="white", size=0.5) +
        geom_point(aes(col=annotation[samp1]), size=0.455, alpha=0.75) +
        geom_point(aes(x=x_coord[samp2], y=y_coord[samp2]),col="black", size=1.1, data = df[samp2,])+
        geom_point(aes(x=x_coord[samp2], y=y_coord[samp2], col =annotation[samp2]), size=0.455, alpha=0.75, data = df[samp2,])+
        scale_color_manual(values=colours) + themes +
        guides(colour = guide_legend(override.aes = list(alpha = 1, size=3)))
    }
  }else{  #plot all celltypes 
    out_plot <- ggplot(df[samp1,], aes(x=x_coord[samp1], y=y_coord[samp1])) + 
      geom_point(col="black", size=0.9) +
      geom_point(col="white", size=0.5) +
      geom_point(aes(col=annotation[samp1]), size=0.455, alpha=0.75) +
      scale_color_manual(values=colours) + themes +
      guides(colour = guide_legend(override.aes = list(alpha = 1, size=3)))
  }
  
  return(out_plot)
}


make_log_plot <- function(sce, feature, df, order=FALSE) {
  
  df$feature <- as.numeric(logcounts(sce[feature,]))
  
  if(order) {
    
    ggplot(df[order(df$feature),], aes(x=UMAP1, y=UMAP2, color=feature)) + geom_point(size=1, alpha=0.7) +
      theme_classic() + scale_color_viridis(feature) + ggtitle(paste(feature)) 
    
  } else {
    
    ggplot(df[sample(1:nrow(df)),], aes(x=UMAP1, y=UMAP2, color=feature)) + geom_point(size=1, alpha=0.7) +
      theme_classic() + scale_color_viridis(feature) + ggtitle(paste(feature)) 
    
  }
  
}

make_ind_plot <- function(sce, feature, df) {
  
  df$feature <- as.numeric(logcounts(sce[feature,]))
  df$feature <- df$feature>0
  
  ggplot(df[order(df$feature),], aes(x=UMAP1, y=UMAP2, color=feature)) + geom_point(size=1, alpha=0.7) +
    theme_classic() + scale_color_manual(values=c("TRUE"="red", "FALSE"="lightgrey"), name=paste0(feature)) 
  
}


find_ranks <- function(markers_tmp, gene_sets, go_term){
  
  gene_set <- gene_sets[[go_term]]
  gene_set <- intersect(rownames(markers_tmp), gene_set)
  signs <- sign(markers_tmp$summary.logFC) 
  markers_tmp$Rank <- log10(markers_tmp$FDR)*-1*signs
  markers_tmp$Rank <- rank(markers_tmp$Rank)
  tmp <- markers_tmp[gene_set,]
  data.frame(GO=go_term, gene=rownames(tmp), Rank=tmp$Rank)
  
}


make_log_map <- function(sce, feature, df, order=FALSE) {
  df$feature <- as.numeric(logcounts(sce[feature,]))
  if(order) {
    ggplot(df[order(df$feature),], aes(x=x_centroid, y=y_centroid, color=feature)) + geom_point(size=1, alpha=0.7) +
      theme_classic() + scale_color_viridis(feature) + ggtitle(paste(feature))
  } else {
    ggplot(df[sample(1:nrow(df)),], aes(x=x_centroid, y=y_centroid, color=feature)) + geom_point(size=1, alpha=0.7) +
      theme_classic() + scale_color_viridis(feature) + ggtitle(paste(feature))
  }
}

