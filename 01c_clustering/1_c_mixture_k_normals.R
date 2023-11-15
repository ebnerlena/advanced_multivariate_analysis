# Mixture of k multivarite normal distributions
# by EM algorithm

library(mvtnorm)

# iris data
?iris
head(iris)
plot(iris[,1:4],col=iris[,5])

k <- 3 # k is assume to b known

# observed data
x <- iris[,1:4]
n <- dim(x)[1]
d <- dim(x)[2]

# 
max.iter <- 40
eps.mu <- 1e-10
eps.Sigma <- 1e-10
eps.p <- 1e-10
eps.ll <- 1e-10

# initial parameters
# with mean vector µj mu and variance matrix Σj sigma
p.0 <- rep(1/k,k)
#mu <- vector("list",k)
#mu.0 <- array(0,dim=c(d,k))
mu.0 <- t(x[sample(1:n,k),])
#Sigma <- vector("list",k)
Sigma.0 <- array(0,dim=c(d,d,k))

for (j in (1:k)) Sigma.0[,,j] <- var(x)

iter <- 0 

ll.EY.0 <- sapply(1:k,function(j,x,mu,Sigma,p){
  p[j]*dmvnorm(x,mean=mu[,j],sigma=Sigma[,,j])
},x,mu.0,Sigma.0,p.0)
ll.0 <- log(prod( apply(ll.EY.0,1,sum)))
stop.criteria <- FALSE

print("  Iter       log.lik        p[1] to p[k]")

while (!stop.criteria){
  print( c( iter, ll.0, p.0) ) 
  iter <- iter+1
  # E step: Estimation
  k <- length(p.0)
  E.Y <- ll.EY.0 / apply(ll.EY.0,1,sum)  
  # M step: Maximization
  p.1 <- apply(E.Y,2,mean)
  mu.1 <- sapply(1:k,function(j,x,E.Y){
    apply(x*E.Y[,j],2,sum)/sum(E.Y[,j])
  },x,E.Y)
  Sigma.1 <- sapply(1:k,function(j,x,E.Y,mu){
    n <- dim(x)[1]
    d <- dim(x)[2]
    x.c <- as.matrix(x)-matrix(mu[,j],ncol=d,nrow=n,byrow=TRUE)
    t(x.c)%*%(x.c*E.Y[,j])/sum(E.Y[,j])
  },x,E.Y,mu.1)
  Sigma.1 <- array(Sigma.1,dim=c(d,d,k))
  # Computing the log-likelihood value
  ll.EY.1 <- sapply(1:k,function(j,x,mu,Sigma,p){
    p[j]*dmvnorm(x,mean=mu[,j],sigma=Sigma[,,j])
  },x,mu.1,Sigma.1,p.1)
  ll.1 <- log(prod( apply(ll.EY.1,1,sum)))
  # checking stopping criteria
  stop.criteria <- (
    (iter==max.iter) |
    (sum((mu.1-mu.0)^2)/prod(dim(mu.1)) <=  eps.mu) |
    (sum((Sigma.1-Sigma.0)^2)/prod(dim(Sigma.1)) <=  eps.Sigma) |
    (sum((p.1-p.0)^2)/length(p.1) <=  eps.p) |
    ((ll.1-ll.0)^2 <= eps.ll)
  )
  if (!stop.criteria){
    mu.0 <- mu.1
    Sigma.0 <- Sigma.1
    p.0 <- p.1
    ll.EY.0 <- ll.EY.1
    ll.0 <- ll.1
  }
}

print( c( iter, ll.1, p.1) ) 

# output 

pred.class <- apply(E.Y,1,function(a){which.max(a)})
print(table(iris$Species,pred.class))

op <- par(mfrow=c(2,1))
plot(E.Y[,1],col="grey",pch=19,cex=1.5)
points(E.Y[,2],col="red",pch=19,cex=1)
points(E.Y[,3],col="black",pch=19,cex=.5)

add.bp <- c(F,T,T)
col.bp <- c("yellow","orange","red")
for (j in 1:3){
  boxplot(E.Y[,j]~iris$Species, 
          boxwex =.2, at=1:3 + .2*(j-2), 
          col=col.bp[j], add=add.bp[j])
}
par(op)

# library(compositions)
# plot(acomp(E.Y),col=iris$Species)
