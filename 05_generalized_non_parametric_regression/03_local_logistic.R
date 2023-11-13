# local.logistic.R
# Illustration of local logistic regression fit
#
# Pedro Delicado

# Plots in the slides have been done with this seed:
# set.seed(2)

set.seed(2)

dev.set(2)

beta0=0
beta1=1
k=4
s=.5

n<-400
x<-sort(rnorm(n))
l.x <- beta0 + beta1*x
theta.x <- l.x + k*x*dnorm(x,m=0,sd=s)
m.x <- 1/(1+exp(-theta.x))
logit.x <- 1/(1+exp(-l.x))

y <- rbinom(n,1,m.x)


# grid
nt<-101
t<-seq(-3.5,3.5,length=nt)
l.t <- beta0 + beta1*t
theta.t <- l.t + k*t*dnorm(t,m=0,sd=s)
m.t <- 1/(1+exp(-theta.t))
logit.t <- 1/(1+exp(-l.t))

op <- par(mfrow=c(1,2))

plot(t,m.t,xlim=c(-3.5,3.5),ylim=c(0,1),type="l")
lines(t,logit.t,col=2)
points(x,y,col=4,pch=3)

plot(t,theta.t,xlim=c(-3.5,3.5),type="l")
lines(t,l.t,col=2)

par(op)

#########

# Estimation by local likelihood at 3 points "t"
if (dev.set(2)!=2) windows()
plot(t,m.t,xlim=c(-3.5,3.5),ylim=c(0,1),type="l")
lines(t,logit.t,col=2)
points(x,y,col=4,pch=3)

if (dev.set(4)!=4) windows()
plot(t,theta.t,xlim=c(-3.5,3.5),type="l")
lines(t,l.t,col=2)

h <- .35
hat.theta.t <- 0*t
hat.m.t <- 0*t
for (i in (1:length(t))){
  ti <- t[i]
  w.ti <- dnorm(x,m=ti,sd=h)
  glm.ti <- glm(y~x,weights=w.ti,family="binomial")
  b0<-glm.ti$coefficients[1]
  b1<-glm.ti$coefficients[2]
  hat.theta.t[i] <- b0 + b1*ti
  hat.m.t[i] <- 1/(1+exp(-hat.theta.t[i]))
}
dev.set(2)
lines(t,hat.m.t,col=6,lwd=2)

dev.set(4)
lines(t,hat.theta.t,col=6,lwd=2)

tt <- c(-1,0,1.5)
for (i in (1:length(tt))){
  ti <- tt[i]
  w.ti <- dnorm(x,m=ti,sd=h)
  glm.ti <- glm(y~x,weights=w.ti,family="binomial")
  b0<-glm.ti$coefficients[1]
  b1<-glm.ti$coefficients[2]
  ti.h <- c(ti-h,ti,ti+h)
  hat.theta.ti <- b0 + b1*ti.h
  hat.m.ti <- 1/(1+exp(-hat.theta.ti))
  
  hat.l.t <- b0 + b1*t
  hat.logit.t <- 1/(1+exp(-hat.l.t))
  
  dev.set(2)
  abline(v=ti,col=8,lty=2)
  points(x,.15*w.ti/max(w.ti),col=8,pch=0)
  lines(ti.h,hat.m.ti,col=3,lwd=2)
  lines(t,hat.logit.t,col=3,lty=3,lwd=2)
  points(ti.h[2],hat.m.ti[2],col=3,pch=19)
  points(x,y,col=4,pch=3)
  
  dev.set(4)
  abline(v=ti,col=8,lty=2)
  lines(ti.h,hat.theta.ti,col=3,lwd=2)
  lines(t,hat.l.t,col=3,lty=3,lwd=2)
  points(ti.h[2],hat.theta.ti[2],col=3,pch=19)
}

dev.set(2)
legend("right",c("m(t)","logit generating m(t)", 
                       "hat{m}(t)", "local logistic fits", "original data"),
       lty=c(1,1,1,3,0), 
       pch=c(NA,NA,NA,19,3),
       col=c(1,2,6,3,4),
       lwd=c(1,1,2,1,NA))

dev.set(4)
legend("bottomright",c("theta(t)","line generating theta(t)", 
                       "hat{theta}(t)", "local linear fits"),
       lty=c(1,1,1,3), 
       pch=c(NA,NA,NA,19),
       col=c(1,2,6,3),
       lwd=c(1,1,2,1,NA))

