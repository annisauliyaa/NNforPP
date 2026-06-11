library(spatstat)
spatstat.options(npixel=256)

load("bubble.RData") #read as BF
df <- BF

df$patterns <- lapply(df$patterns, unmark)
df

# observation window
# W <- owin()
# number of processes
m <- 9
# number of replications for each process
Tn <- 10

df_group <- split(df$patterns, df$group)

lapply(df_group, function(o){ mean(unlist(lapply(o, npoints))) })

process <- c("5_5", "5_8", "5_10", "10_5", "10_8", 
             "10_10", "15_5", "15_8", "15_10")
par(mfrow=c(m, Tn), mar=c(0.5, 0.5, 1.25, 0.5))
for (i in 1:m)
{
  for (j in 1:Tn)
  {
    plot(df_group[[process[i]]][[j]], main=paste(process[i], j), cex=0.25)
  }
}
dev.copy2pdf(file="bubblepatterns.pdf", width=10, height=11)


library(feather)
dir.create("~/bubPaterns")
for (i in 1:m)
{
  for (j in 1:Tn)
  {
    df <- df_group[[process[i]]][[j]]
    df <- data.frame(x=df$x, y=df$y)
    fname <- paste0("~/bubPaterns/", process[i], j, ".feather")
    write_feather(df, fname)
  }
}

nx <- 100
ny <- 100
cellCounts <- vector("list", length=m)
out <- c()
for (i in 1:m)
{
  Cs <- vector("list", length=Tn)
  for (j in 1:Tn)
  {
    
    Cs[[j]] <- quadratcount(df_group[[process[i]]][[j]], nx=nx, ny=ny)
    out <- c(out, t(Cs[[j]]))
  }
  cellCounts[[i]] <- Cs
}

write.table(out, file="bubdata.txt", row.names = FALSE,
            col.names = FALSE)

