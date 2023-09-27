# Mixture of 2 univariate normal distributions
# by EM algorithm

library(sm)
# iris data
?iris
#names(iris)
#[1] "Sepal.Length" "Sepal.Width"  "Petal.Length" "Petal.Width"  "Species"     
head(iris)
plot(iris[,1:4],col=iris[,5])

k <- 2 # k is assume to be known

# observed data
x <- iris[,3] # Petal.Length
hist(x)
n <- length(x)

# stop criterions (some thresholds of the algorithm)
max.iter <- 40
eps.mu <- 1e-10
eps.sigma <- 1e-10
eps.p <- 1e-10
eps.ll <- 1e-10

# initial parameters
p.0 <- c(1/2,1/2)
#mu <- vector("list",k)
#mu.0 <- array(0,dim=c(d,k))
mu.0 <- sample(x,2)
sigma.0 <- rep(sd(x),2)

iter <- 0 

ll.EY.0 <- sapply(1:k,
                  function(j,x,mu,sigma,p){
                    p[j]*dnorm(x,mean=mu[j],sd=sigma[j])
                  },
                  x,mu.0,sigma.0,p.0)
ll.0 <- log(prod( apply(ll.EY.0,1,sum)))
stop.criteria <- FALSE

print("  Iter   log.lik    p[1]     p[2]     mu[1]    mu[2]    sigma[1] sigma[2]")

while (!stop.criteria){
  print( c( iter, ll.0, p.0, mu.0, sigma.0), digits=3) 
  iter <- iter+1
  # E step: Estimation
  k <- length(p.0)
  E.Y <- ll.EY.0 / apply(ll.EY.0,1,sum)  
  # M step: Maximization
  p.1 <- apply(E.Y,2,mean)
  mu.1 <- sapply(1:k,function(j,x,E.Y){
    sum(x*E.Y[,j])/sum(E.Y[,j])
  },x,E.Y)
  sigma.1 <- sapply(1:k,function(j,x,E.Y,mu){
    n <- length(x)
    x.c <- x-mu[j]
    sqrt( sum(E.Y[,j] * x.c^2)/sum(E.Y[,j]))
  },x,E.Y,mu.1)
  # Computing the log-likelihood value
  ll.EY.1 <- sapply(1:k,function(j,x,mu,sigma,p){
    p[j]*dnorm(x,mean=mu[j],sd=sigma[j])
  },x,mu.1,sigma.1,p.1)
  ll.1 <- log(prod( apply(ll.EY.1,1,sum)))
  # checking stopping criteria
  stop.criteria <- (
    (iter==max.iter) |
      (sum((mu.1-mu.0)^2)/length(mu.1) <=  eps.mu) |
      (sum((sigma.1-sigma.0)^2)/length(sigma.1) <=  eps.sigma) |
      (sum((p.1-p.0)^2)/length(p.1) <=  eps.p) |
      ((ll.1-ll.0)^2 <= eps.ll)
  )
  if (!stop.criteria){
    mu.0 <- mu.1
    sigma.0 <- sigma.1
    p.0 <- p.1
    ll.EY.0 <- ll.EY.1
    ll.0 <- ll.1
  }
}

print( c( iter, ll.1, p.1, mu.1, sigma.1), digits=3 ) 

# output 

pred.class <- apply(E.Y,1,function(a){which.max(a)})
print(table(iris$Species,pred.class))

op <- par(mfrow=c(2,2))
plot(E.Y[,1],col="grey",pch=19,cex=1.5,ylim=c(0,1))
points(E.Y[,2],col="red",pch=19,cex=1)

add.bp <- c(F,T,T)
col.bp <- c("yellow","orange","red")
for (j in 1:2){
  boxplot(E.Y[,j]~iris$Species, 
          boxwex =.2, at=1:3 + .2*(j-2), 
          col=col.bp[j], add=add.bp[j])
}


hist(x,freq=FALSE,ylim=c(0,.8))
plot(function(x){p.1[1]*dnorm(x,mu.1[1],sigma.1[1]) + p.1[2]*dnorm(x,mu.1[2],sigma.1[2])},
      xlim=range(x),col=2,add=TRUE,lwd=2)

aux <- sm.density(x,method="sj",ylim=c(0,.8),lwd=2)
abline(h=0,col=8)
plot(function(x){p.1[1]*dnorm(x,mu.1[1],sigma.1[1]) + p.1[2]*dnorm(x,mu.1[2],sigma.1[2])},
     xlim=range(aux$eval.points),n=201, col=2,add=TRUE,lwd=2)
par(op)

