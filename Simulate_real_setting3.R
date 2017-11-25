rm(list=ls())
library(MASS)
library(Rcpp)
library(bayeslm)

# setwd("/home/xin/ownCloud/Research/STAT548/Second paper___Paul/simulation")
setwd("C:/Users/dingx/ownCloud/Research/STAT548/Second paper___Paul/simulation")
sourceCpp("BayesRegLM.cpp")

#3.2 setting 1
# setups
Setting_name = "_setting3_"
prior = "ridge" #"ridge" or "horseshoe"
finalStep = 2 # 1 means without final step; 2 means with final step, sample alpha from normal; 3 means with final step, use mean
nSim = 500
n = 100
p = 30 #numbers of columns of X (exclude Z)
k = 3
kappa2 = 0.05
phi2 = 0.05
sigma2_nu = 0.9
rho2_grid = c(0.1, 0.3, 0.5, 0.7, 0.9)
# rho2_grid = c(0.1,0.3)
sigma2_ep_grid = 1 - rho2_grid
alpha_grid = sqrt(kappa2/(1-rho2_grid))

#number of samplers from posterior distribution
nsamps = 5000

#file name for Rdata
filename = paste0("Sim_real_",prior,Setting_name,"nSim",nSim,"_n",n,"_p",p,"_k",k,"_kappa2_",kappa2,
                  "_phi2_",phi2,"_sigma2_nu_",sigma2_nu,".Rdata",seq="")

#initialization
#arrays to store estimated alpha and its standard error
fit1_coeff=fit2_coeff=fit3_coeff=fit4_coeff=fit5_coeff = 
  array(0,c(nSim, 2, length(rho2_grid)))
bias1=bias2=bias3=bias4=bias5 = 
  array(0, c(nSim, length(rho2_grid))) #store bias
MSE1=MSE2=MSE3=MSE4=MSE5 = 
  array(0, c(nSim, length(rho2_grid))) #store MSE
CI1=CI2=CI3=CI4=CI5 = array(0,c(nSim,2,length(rho2_grid))) #store CI
CP1=CP2=CP3=CP4=CP5=array(0, c(nSim, length(rho2_grid))) #store Coverage Probability
IL1=IL2=IL3=IL4=IL5=array(0, c(nSim, length(rho2_grid))) #store interval length
cutoff_point=qnorm(0.975,0,1)


for (nrho in 1:length(rho2_grid)){
  for (nn in 1:nSim){
    print(c(nn,nrho))
    #-------------------------------------------
    #data generation
    set.seed(nn*10+nrho)
    rho2 = rho2_grid[nrho] #rho2
    alpha = alpha_grid[nrho]
    # alpha = sqrt (kappa2 / (1 - rho2)) # alpha
    sigma2_ep = sigma2_ep_grid[nrho] #sigma2_ep
    
    #generate betaC and betaD
    betaC_ini = rep(0, p) #inital values of betaC
    betaC_ini[1:(2*k)] = rep(1, 2*k)
    betaD_ini = rep(0, p) #inital values of betaD
    betaD_ini[(k+1):(3*k)] = rnorm(2*k,0,1)
    Sc = sqrt(rho2/(2*k)) #scale prameter for betaC_ini
    Sd = sqrt(phi2/sum(betaD_ini^2))
    betaC = betaC_ini * Sc #scale
    betaD = betaD_ini * Sd #scale
    beta = betaD - alpha * betaC
    
    #generate predictors and Y
    X = mvrnorm(n = n,mu = rep(0, p), Sigma = diag(p)) #generate X
    Z = X %*% betaC + rnorm(n,0,sigma2_ep) #generate Z by selection Eq.
    Y = alpha * Z + X %*% beta + rnorm(n, 0, sigma2_nu)
    indx_beta_nonzero = (beta!=0)
    Var_all = cbind(Z,X)
    
    #-------------------------------------------
    #Fit models
    #new approach:fit1
    if (prior=="horseshoe"){
      fit1 = BayesRegLMTEE2M(matrix(Y,nrow = n), Var_all, nsamps = nsamps, prior = 1,
                             sigma2_nu = sigma2_nu, sigma2_ep = sigma2_ep, HyperShr = 1,
                             finalStep = finalStep);
    }else if(prior=="ridge"){
      fit1 = BayesRegLMTEE2M(matrix(Y,nrow = n), Var_all, nsamps = nsamps, prior = 2, 
                             sigma2_nu = sigma2_nu, sigma2_ep = sigma2_ep, HyperShr = 1,
                             finalStep = finalStep);
    }
    fit1_coeff[nn,1,nrho] = fit1$alpha[1]
    fit1_coeff[nn,2,nrho] = fit1$alpha_SD[1]
    bias1[nn,nrho]=fit1_coeff[nn,1,nrho]-alpha
    CI1[nn,1,nrho]=fit1_coeff[nn,1,nrho]-fit1_coeff[nn,2,nrho]*cutoff_point
    CI1[nn,2,nrho]=fit1_coeff[nn,1,nrho]+fit1_coeff[nn,2,nrho]*cutoff_point
    IL1[nn,nrho] = 2*fit1_coeff[nn,2,nrho]*cutoff_point
    
    #Ordinary OLS:fit2
    fit2=lm(Y~Var_all-1)#exclude intercept
    fit2_coeff[nn,1,nrho]=summary(fit2)$coefficients[1,1]
    fit2_coeff[nn,2,nrho]=summary(fit2)$coefficients[1,2]
    bias2[nn,nrho]=fit2_coeff[nn,1,nrho]-alpha
    CI2[nn,1,nrho]=fit2_coeff[nn,1,nrho]-fit2_coeff[nn,2,nrho]*cutoff_point
    CI2[nn,2,nrho]=fit2_coeff[nn,1,nrho]+fit2_coeff[nn,2,nrho]*cutoff_point
    IL2[nn,nrho] = 2*fit2_coeff[nn,2,nrho]*qnorm(0.975,0,1)
    
    #Naive Regularization:fit3
    if (prior=="horseshoe"){
      fit3 = BayesRegLMTEE(matrix(Y,nrow = n),Var_all, nsamps = nsamps, prior = 1,
                           sigma2 = sigma2_nu, HyperShr = 1);
    }else if (prior == "ridge"){
      fit3 = BayesRegLMTEE(matrix(Y,nrow = n),Var_all, nsamps = nsamps, prior = 2,
                           sigma2 = sigma2_nu, HyperShr = 1);
    }
    fit3_coeff[nn,1,nrho] = fit3$beta[1]
    fit3_coeff[nn,2,nrho] = fit3$beta_SD[1]
    bias3[nn,nrho]=fit3_coeff[nn,1,nrho]-alpha
    CI3[nn,1,nrho]=fit3_coeff[nn,1,nrho]-fit3_coeff[nn,2,nrho]*cutoff_point
    CI3[nn,2,nrho]=fit3_coeff[nn,1,nrho]+fit3_coeff[nn,2,nrho]*cutoff_point
    IL3[nn,nrho] = 2*fit3_coeff[nn,2,nrho]*cutoff_point
    
    #Oracle OLS:fit4
    fit4=lm(Y~Z+X[,indx_beta_nonzero]-1)
    fit4_coeff[nn,1,nrho]=summary(fit4)$coefficients[1,1]
    fit4_coeff[nn,2,nrho]=summary(fit4)$coefficients[1,2]
    bias4[nn,nrho]=fit4_coeff[nn,1,nrho]-alpha
    CI4[nn,1,nrho]=fit4_coeff[nn,1,nrho]-fit4_coeff[nn,2,nrho]*cutoff_point
    CI4[nn,2,nrho]=fit4_coeff[nn,1,nrho]+fit4_coeff[nn,2,nrho]*cutoff_point
    IL4[nn,nrho] = 2*fit4_coeff[nn,2,nrho]*cutoff_point
    
    #Naive Regularization by bayeslm: fit5
    fit5=bayeslm(Y~Var_all, prior=prior, penalize=c(0,rep(1,p)),
                 icept=FALSE, N=nsamps, verb=FALSE, sigma = sigma2_nu)
    fit5_coeff[nn,1,nrho]=mean(as.matrix(fit5$beta)[,2])#why setting icept=FALSE is useless?
    fit5_coeff[nn,2,nrho]=sd(as.matrix(fit5$beta)[,2])
    bias5[nn,nrho]=fit5_coeff[nn,1,nrho]-alpha
    CI5[nn,1,nrho]=fit5_coeff[nn,1,nrho]-fit5_coeff[nn,2,nrho]*cutoff_point
    CI5[nn,2,nrho]=fit5_coeff[nn,1,nrho]+fit5_coeff[nn,2,nrho]*cutoff_point
    IL5[nn,nrho] = 2*fit5_coeff[nn,2,nrho]*cutoff_point
  }
}

results = array(0, c(25,4))
for (i in 1:5){
  alpha = alpha_grid[i]
  results [(5*(i-1)+1):(5*i),] = 
    cbind(c(mean(bias1[,i]),mean(bias2[,i]),mean(bias3[,i]),mean(bias4[,i]),mean(bias5[,i])),
          c(mean(CI1[,1,i]<alpha & CI1[,2,i]>alpha),mean(CI2[,1,i]<alpha & CI2[,2,i]>alpha),
            mean(CI3[,1,i]<alpha & CI3[,2,i]>alpha),mean(CI4[,1,i]<alpha & CI4[,2,i]>alpha),
            mean(CI5[,1,i]<alpha & CI5[,2,i]>alpha)),
          c(mean(IL1[,i]),mean(IL2[,i]),mean(IL3[,i]),mean(IL4[,i]),mean(IL5[,i])),
          c(mean((fit1_coeff[,1,i]-alpha)^2),mean((fit2_coeff[,1,i]-alpha)^2),
            mean((fit3_coeff[,1,i]-alpha)^2),mean((fit4_coeff[,1,i]-alpha)^2),
            mean((fit5_coeff[,1,i]-alpha)^2)))
}


# results=round(results,digits=4)
# library(XLConnect)
# library(xlsx)
# wb <- XLConnect::loadWorkbook("tmp.xlsx", create = TRUE)
# sheetname="Sheet1"
# XLConnect::writeWorksheet(wb,results,sheetname,startRow = 1, startCol = 1, header = FALSE)
# XLConnect::saveWorkbook(wb)


save.image(filename)