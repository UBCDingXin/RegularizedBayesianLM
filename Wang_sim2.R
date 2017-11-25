rm(list=ls())
library(MASS)
library(Rcpp)
library(bayeslm)

# setwd("/home/xin/ownCloud/Research/STAT548/Second paper___Paul/simulation")
setwd("C:\\Users\\dingx\\ownCloud\\Research\\STAT548\\Second paper___Paul\\simulation")
sourceCpp("BayesRegLM.cpp")

#First simulation
#simulation setups
prior = "horseshoe" #"horseshoe" or ridge
nsamps = 5000
nSim=1000#number of simulations
n=1000#sample size
p=58#the number of all regressors
rho=0.7#cov of Z and X1
Sigma_ZXs=diag(8)#cov of Z, X1, ..., X7
for (i in 1:8){
  for (j in 1:8){
    if (i!=j){Sigma_ZXs[i,j]=rho^{i+j-2}}
  }
}
Sigma_all=diag(p)#cov of Z and all Xs
Sigma_all[1:8,1:8]=Sigma_ZXs

alpha=0.1#coefficients
beta=rep(0.1,14)

#arrays to store estimated alpha and its standard error
fit1_coeff=fit2_coeff=fit3_coeff=fit4_coeff=fit5_coeff=array(0,c(nSim,2))
bias1=bias2=bias3=bias4=bias5=c() #store bias
MSE1=MSE2=MSE3=MSE4=MSE5=c() #store MSE
CI1=CI2=CI3=CI4=CI5=array(0,c(nSim,2)) #store CI
CP1=CP2=CP3=CP4=CP5=c() #store Coverage Probability
IL1=IL2=IL3=IL4=IL5=c() #store interval length
cutoff_point=qnorm(0.975,0,1)
# cutoff_point=1

for (nn in 1:nSim){
  print(nn)
  #-------------------------------------------
  #data generation
  set.seed(nn)
  #generate Z, X1, X2, and other Xs from multivariate normal
  Var_all=mvrnorm(n=n,mu=rep(0,p),Sigma=Sigma_all)
  Z=Var_all[,1]
  Xs=Var_all[,2:15]
  X_all=Var_all[,2:p]
  #generate response Y
  err=rnorm(n=n,mean=0,sd=1)
  Y=alpha*Z+Xs%*%beta+err
  #-------------------------------------------
  #Fit models
  #new approach:fit1
  if (prior=="horseshoe"){
    fit1 = BayesRegLMTEE2M(matrix(Y,nrow = n),Var_all, nsamps = nsamps, prior = 1, finalStep = finalStep);
  }else if(prior=="ridge"){
    fit1 = BayesRegLMTEE2M(matrix(Y,nrow = n),Var_all, nsamps = nsamps, prior = 2, finalStep = finalStep);
  }
  fit1_coeff[nn,1] = fit1$alpha[1]
  fit1_coeff[nn,2] = fit1$alpha_SD[1]
  bias1[nn]=fit1_coeff[nn,1]-alpha
  CI1[nn,1]=fit1_coeff[nn,1]-fit1_coeff[nn,2]*cutoff_point
  CI1[nn,2]=fit1_coeff[nn,1]+fit1_coeff[nn,2]*cutoff_point
  IL1[nn] = 2*fit1_coeff[nn,2]*cutoff_point
  
  #Ordinary OLS:fit2
  fit2=lm(Y~Var_all-1)#exclude intercept
  fit2_coeff[nn,1]=summary(fit2)$coefficients[1,1]
  fit2_coeff[nn,2]=summary(fit2)$coefficients[1,2]
  bias2[nn]=fit2_coeff[nn,1]-alpha
  CI2[nn,1]=fit2_coeff[nn,1]-fit2_coeff[nn,2]*cutoff_point
  CI2[nn,2]=fit2_coeff[nn,1]+fit2_coeff[nn,2]*cutoff_point
  IL2[nn] = 2*fit2_coeff[nn,2]*cutoff_point
  
  #Naive Regularization:fit3
  if (prior=="horseshoe"){
    fit3 = BayesRegLMTEE(matrix(Y,nrow = n),Var_all, nsamps = nsamps, prior = 1);
  }else if (prior == "ridge"){
    fit3 = BayesRegLMTEE(matrix(Y,nrow = n),Var_all, nsamps = nsamps, prior = 2);
  }
  fit3_coeff[nn,1] = fit3$beta[1]
  fit3_coeff[nn,2] = fit3$beta_SD[1]
  bias3[nn]=fit3_coeff[nn,1]-alpha
  CI3[nn,1]=fit3_coeff[nn,1]-fit3_coeff[nn,2]*cutoff_point
  CI3[nn,2]=fit3_coeff[nn,1]+fit3_coeff[nn,2]*cutoff_point
  IL3[nn] = 2*fit3_coeff[nn,2]*cutoff_point
  
  #Oracle OLS:fit4
  fit4=lm(Y~Var_all[,1:15]-1)
  fit4_coeff[nn,1]=summary(fit4)$coefficients[1,1]
  fit4_coeff[nn,2]=summary(fit4)$coefficients[1,2]
  bias4[nn]=fit4_coeff[nn,1]-alpha
  CI4[nn,1]=fit4_coeff[nn,1]-fit4_coeff[nn,2]*cutoff_point
  CI4[nn,2]=fit4_coeff[nn,1]+fit4_coeff[nn,2]*cutoff_point
  IL4[nn] = 2*fit4_coeff[nn,2]*cutoff_point
  
  #Naive Regularization by bayeslm: fit5
  fit5=bayeslm(Y~Var_all, prior=prior, penalize=c(0,rep(1,p-1)),
               icept=FALSE, N=nsamps, verb=FALSE)
  fit5_coeff[nn,1]=mean(as.matrix(fit5$beta)[,2])#why setting icept=FALSE is useless?
  fit5_coeff[nn,2]=sd(as.matrix(fit5$beta)[,2])
  bias5[nn]=fit5_coeff[nn,1]-alpha
  CI5[nn,1]=fit5_coeff[nn,1]-fit5_coeff[nn,2]*cutoff_point
  CI5[nn,2]=fit5_coeff[nn,1]+fit5_coeff[nn,2]*cutoff_point
  IL5[nn] = 2*fit5_coeff[nn,2]*cutoff_point
}
#avg bias
c(mean(bias1),mean(bias2),mean(bias3),mean(bias4),mean(bias5))

#avg sd
c(mean(fit1_coeff[,2]),mean(fit2_coeff[,2]),
  mean(fit3_coeff[,2]),mean(fit4_coeff[,2]))

#MSE
MSE1 = mean((fit1_coeff[,1]-alpha)^2) 
MSE2 = mean((fit2_coeff[,1]-alpha)^2) 
MSE3 = mean((fit3_coeff[,1]-alpha)^2) 
MSE4 = mean((fit4_coeff[,1]-alpha)^2) 
MSE5 = mean((fit5_coeff[,1]-alpha)^2) 
c(MSE1,MSE2,MSE3,MSE4,MSE5)

#coverage prob
c(mean(CI1[,1]<alpha & CI1[,2]>alpha),mean(CI2[,1]<alpha & CI2[,2]>alpha),
  mean(CI3[,1]<alpha & CI3[,2]>alpha),mean(CI4[,1]<alpha & CI4[,2]>alpha),
  mean(CI5[,1]<alpha & CI5[,2]>alpha))

#avg IL
c(mean(IL1),mean(IL2),mean(IL3),mean(IL4),mean(IL5))

results = cbind(c(mean(bias1),mean(bias2),mean(bias3),mean(bias4),mean(bias5)),
                c(mean(CI1[,1]<alpha & CI1[,2]>alpha),mean(CI2[,1]<alpha & CI2[,2]>alpha),
                  mean(CI3[,1]<alpha & CI3[,2]>alpha),mean(CI4[,1]<alpha & CI4[,2]>alpha),
                  mean(CI5[,1]<alpha & CI5[,2]>alpha)),
                c(mean(IL1),mean(IL2),mean(IL3),mean(IL4),mean(IL5)),
                c(MSE1,MSE2,MSE3,MSE4,MSE5))

# results=round(results,digits=4)
# Sys.setenv(JAVA_HOME='C:/Program Files/Java/jdk1.8.0_131/jre')
# library(XLConnect)
# library(xlsx)
# wb <- XLConnect::loadWorkbook("tmp.xlsx", create = TRUE)
# sheetname="Sheet1"
# XLConnect::writeWorksheet(wb,results,sheetname,startRow = 1, startCol = 1, header = FALSE)
# XLConnect::saveWorkbook(wb)

# save.image("Wang_sim2_horseshoe.Rdata")

file = paste0("Wang_sim2_",prior,".Rdata",seq="")
save.image(file)
