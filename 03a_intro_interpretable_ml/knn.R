################ k-nn regression
knn_regr<- function(x, y, t=NULL, k=3, 
                    dist.method = "euclidean"){
  nx <- length(y)
  if (is.null(t)){
    t<- as.matrix(x)
  }else{
    t<-as.matrix(t)
  }
  nt <- dim(t)[1]
  Dtx <- as.matrix( dist(rbind(t,as.matrix(x)),
                         method = dist.method) )
  Dtx <- Dtx[1:nt,nt+(1:nx)]
  mt <- numeric(nt)
  for (i in 1:nt){
    d_t_x <- Dtx[i,]
    d_t_x_k <- sort(d_t_x,partial=k)[k]
    N_t_k <- unname( which(d_t_x <= d_t_x_k) )
    mt[i]=mean(y[N_t_k])
  }
  return(mt)
}	

set.seed(1234)

n <- 200; sd_eps <- .05
x <- sort(2*runif(n)-1)
mx <- 1-x^2 # regression fuction
eps <- rnorm(n,0,sd_eps) # random noise
y <- mx+eps
plot(x,y,xlim=c(-1,1),ylim=c(-3*sd_eps,1+3*sd_eps), col=8)

k <- n/20
hat_mx <- knn_regr(x,y,k=k)
lines(x,hat_mx,col=2,lwd=2)
title(main=paste0("k-nn regression estimator, k=",k))

#### pdf file
saving_pdf <- FALSE
if (saving_pdf) {
  pdf("knn_example.pdf", width=6, height=4)
  plot(x,y,xlim=c(-1,1),ylim=c(-3*sd_eps,1+3*sd_eps),col=8)
  lines(x,hat_mx,col=2,lwd=2)
  title(main=paste0("k-nn regression estimator, k=",k))
  dev.off()
}

################ k-nn classification
knn_class<- function(x, y, t=NULL, k=3, dist.method="euclidean"){
  nx <- length(y)
  classes <- sort(unique(y))
  nclasses <- length(classes)
  if (is.null(t)){
    t <- as.matrix(x)
  }else{
    t <- as.matrix(t)
  }
  nt <- dim(t)[1]
  Dtx <- as.matrix(dist(rbind(t,as.matrix(x)), method=dist.method))
  Dtx <- Dtx[1:nt,nt+(1:nx)]
  hat_probs_t <- matrix(0, nrow = nt, ncol=nclasses)
  hat_y_t <- numeric(nt)
  for (i in 1:nt){
    d_t_x <- Dtx[i,]
    d_t_x_k <- sort(d_t_x,partial=k)[k]
    Ntk <- unname( which(d_t_x <= d_t_x_k) )
    for (j in 1:nclasses){
      hat_probs_t[i,j] <- sum(y[Ntk]==classes[j])/length(Ntk)
    }
    hat_y_t[i] <- classes[which.max(hat_probs_t[i,])]
  }
  return(list(hat_y_t=hat_y_t, hat_probs_t=hat_probs_t))
}	

set.seed(5678)
n <- 200; sd_eps <- .05
x <- matrix(2*runif(2*n)-1, ncol=2)
px <- exp(-x[,1]^2-x[,2]^2) # = 1/2
y <- rbinom(n,size=1,prob = px)
plot(x[,1],x[,2],xlim=c(-1,1),ylim=c(-1,1), col=y+1, asp=1)
abline(h=0,v=0,col=8,lty=3)
lines(sqrt(log(2))*cos(seq(0,2*pi,length=201)), 
      sqrt(log(2))*sin(seq(0,2*pi,length=201)),col=8,lty=2)

k <- n/20
hat_y <- knn_class(x,y,k=k) # with prediction he fills outer circle, original value is inner circle, black = 1, red = 0,
points(x[,1],x[,2], pch=19, cex=.5, col=hat_y$hat_y_t+1)
title(main=paste0("k-nn classification estimator, k=",k,
                  "\n Misclassification rate: ", mean(y!=hat_y$hat_y_t)))

#### pdf file
saving_pdf <- FALSE
if (saving_pdf) {
  pdf("knn_class_example.pdf", width=6, height=6)
  plot(x[,1],x[,2],xlim=c(-1,1),ylim=c(-1,1), col=y+1, asp=1)
  abline(h=0,v=0,col=8,lty=3)
  lines(sqrt(log(2))*cos(seq(0,2*pi,length=201)), 
        sqrt(log(2))*sin(seq(0,2*pi,length=201)),col=8,lty=2)
  
  k <- n/20
  hat_y <- knn_class(x,y,k=k)
  points(x[,1],x[,2], pch=19, cex=.5, col=hat_y$hat_y_t+1)
  title(main=paste0("k-nn classification estimator, k=",k,
                    "\n Misclassification rate: ", mean(y!=hat_y$hat_y_t)))
  dev.off()
}

