//[[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
using namespace Rcpp;
using namespace arma;

inline double LogHorseshoePrior(arma::vec beta, double HyperShr){
  // for prior eval
  // prior: horseshoe; priorType = 1
  double p = (double) beta.n_elem;
  vec beta2 = conv_to<vec>::from(pow((beta/HyperShr) , 2.0));
  double ll = sum(log(log(1.0 + 4.0 / beta2))) - std::log1p(HyperShr) * p;
  return ll;
}

inline double log_stdnormal(arma::vec beta){
  // for prior eval
  // prior: normal; priorType = 2
  double p = (double) beta.n_elem;
  double ll = conv_to<double>::from(-0.5*trans(beta)*beta-0.5*p*std::log1p(2*M_PI));
  return ll;
}

inline double log_normal_density(double x, double mu, double sigma){
  // for global shrinkage parameter update
  // returns log density of normal(mu, sigma)
  double output = -0.5 * log(2.0 * M_PI) - log(sigma) - pow((x - mu), 2) / 2.0 / pow(sigma, 2);
  return(output);
}


/*-------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------*/
// [[Rcpp::export]]
List BayesRegLMTEE(arma::mat Y, arma::mat X, double sigma2 = 1, int priorType = 1,
                      double HyperShr = 1, int nsamps = 5000){
  // Bayesian Regularized Linear Model Treatment Effect Estimation
  // only consider response equation
  
  // dimensions
  int n = X.n_rows;
  int p = X.n_cols;
  
  // variable definition and initialization
  mat XY, XX; // for calculate OLS
  vec beta_hat(p), beta(p), betaPri(p);
  double h = 0; // number of samples been chosen
  mat bsamps(p, nsamps); // matrix to save nsamps posterior samples
  bsamps.fill(0.0);
  vec bmeans(p); // vector to save posterior means
  bmeans.fill(0.0);
  vec bsds(p); // vector to save posterior standard deviation
  bsds.fill(0.0);
  vec zeta(p); // 1st step of gibbs, draw from normal with mean 0 variance sig^2*inv(XX)
  double nu; // 2nd step of gibbs, draw from Uniform(0,1)
  double lli, ll; // ini loglik and loglik in 2nd step of gibb
  double phi, lb, ub; // 3rd step of gibbs, draw from U(0,2pi), lower bound and upper bound
  vec Delta(p), DeltaPri(p); // 4th step of gibbs
  
  //OLS estimate: beta_hat and use it as inital value for beta
  XY = trans(X) * Y;
  XX = trans(X) * X;
  beta_hat = inv(XX) * XY;
  // a initial value of the derivation from the mean
  beta = 1.1 * beta_hat;
  // a initial value of Delta
  Delta = beta - beta_hat;
  
  // loop, stop until get enough posterior samples
  while (h < nsamps){
    // std::cout<<h<<std::endl;
    // 1st step; random generate zeta from normal distribution with mean 0 variance sigma^2*inv(t(X)X)
    zeta = (randn(1, p) * chol(sigma2*inv(XX))).t();
    
    // 2nd step; nu~U(0,1)
    nu = arma::as_scalar(randu(1));
    // initialize log likelihood value
    if (priorType == 1){
      // horseshoe prior
      lli = LogHorseshoePrior(beta.tail(p-1), HyperShr) + log(nu);
    }
    if(priorType == 2){
      // normal prior
      lli = log_stdnormal(beta.tail(p-1)) + log(nu);
    }
    
    // 3nd step; Draw angle from phi~U(0,2*pi)
    phi = arma::as_scalar(randu(1)*2*M_PI);
    lb = phi - 2.0 * M_PI;
    ub = phi;
    
    // 4th step;
    DeltaPri = Delta * cos(phi) + zeta * sin(phi);
    betaPri = beta_hat + DeltaPri;
    
    // 5th step;
    int iter = 0;
    if (priorType == 1){
      ll = LogHorseshoePrior(betaPri.tail(p-1), HyperShr);
    }else if (priorType == 2){
      ll = log_stdnormal(betaPri.tail(p-1));
    }
    while (ll < lli){
      // std::cout<<h<<","<<iter<<std::endl;
      // 5a
      if (phi < 0){
        lb = phi;
      }else{
        ub = phi;
      }
      // 5b
      phi = runif(1, lb, ub)[0];
      // 5c
      DeltaPri = Delta * cos(phi) + zeta * sin(phi);
      betaPri = beta_hat + DeltaPri;
      
      if (priorType == 1){
        ll = LogHorseshoePrior(betaPri.tail(p-1), HyperShr);
      } else if (priorType == 2){
        ll = log_stdnormal(betaPri.tail(p-1));
      }
      
      iter++;
    }
    
    // 6th step;
    Delta = DeltaPri;
    beta = beta_hat + DeltaPri;
    
    // output results
    bsamps.col(h) = beta; // posterior samples
    
    // update the global shrinkage parameter
    if (priorType == 1){
      double HyperShr_tmp = exp(log(HyperShr) + as_scalar(randn(1))*0.05);
      double ratio = exp(LogHorseshoePrior(beta.tail(p-1), sigma2 * HyperShr_tmp)+
                         log_normal_density(HyperShr_tmp, 0.0, 100.0) - 
                         LogHorseshoePrior(beta.tail(p-1), sigma2 * HyperShr)-
                         log_normal_density(HyperShr, 0.0, 100.0)+
                         log(HyperShr_tmp) - log(HyperShr));
      if(as_scalar(randu(1)) < ratio){
        HyperShr = HyperShr_tmp;
      }
    }
    h++; // update h
  }//end while(h<nsamps)
  
  for (int i = 0; i < p; i++){
    bmeans(i) = mean(bsamps.cols(round(nsamps * 0.1),nsamps - 1).row(i)); // posterior mean
    bsds(i) = stddev(bsamps.cols(round(nsamps * 0.1),nsamps - 1).row(i)); // posterior standard deviation
  }
  
  return List::create(_["beta"] = bmeans, _["beta_SD"] = bsds, _["bsamps"] = bsamps);
}


/*-------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------*/
// [[Rcpp::export]]
List BayesRegLMTEE2M(arma::mat Y, arma::mat W, double sigma2_nu = 1, double sigma2_ep = 1, 
                        int priorType = 1, double HyperShr = 1, int nsamps = 5000, int finalStep = 2){
  // Bayesian Regularized Linear Model Treatment Effect Estimation
  // consider both response equation and selection equation
  
  // parameters
  /* priorType: 1. Horseshoe; 2. ridge. 
   * HyperShr: shrinkage. 
   * nsamps: number of samples draw from posterior distribution
   * finalStep: 1. without final step. 2. final step sample from normal. 3. final step use mean
   */
  
  // dimensions
  int n = W.n_rows; //number of observations
  int p = W.n_cols; //number of all predictors (including Z)
  
  // W = [Z,X]
  mat Z(n,1); 
  Z = W.col(0);
  mat X(n,p-1);
  X = W.cols(1,p-1);
  
  // variable definition and initialization
  mat XX, XZ, WW, WY; // for calculate OLS
  double alpha, alpha_hat, alphaPri;
  vec beta(p-1), beta_hat(p-1), betaPri(p-1);
  vec gamma(p-1), gamma_hat(p-1), gammaPri(p-1);
  vec betastar_hat(p); // betastar_hat = [alpha_hat, beta_hat]; OLS of response equation
  vec betaC(p-1), betaCPri(p-1), betaD(p-1), betaDPri(p-1);
  double h = 0; // number of samples have been chosen
  // posterior alpha
  vec asamps(nsamps); // posterior samples for alpha
  asamps.fill(0.0);
  double amean, asd; // posterior mean and sd of alpha
  // posterior betaC
  mat bcsamps(p-1, nsamps); // matrix to save nsamps posterior samples for betaC
  bcsamps.fill(0.0);
  vec bcmeans(p-1); // vector to save posterior means
  bcmeans.fill(0.0);
  vec bcsds(p-1); // vector to save posterior standard deviation
  bcsds.fill(0.0);
  // posterior betaD
  mat bdsamps(p-1, nsamps); // matrix to save nsamps posterior samples for betaD
  bdsamps.fill(0.0);
  vec bdmeans(p-1); // vector to save posterior means
  bdmeans.fill(0.0);
  vec bdsds(p-1); // vector to save posterior standard deviation
  bdsds.fill(0.0);
  
  vec zeta1(p), zeta2(p-1), zeta(2*p-1); // 1st step of gibbs, draw from normal with mean 0 variance sig^2*inv(XX)
  double nu; // 2nd step of gibbs, draw from Uniform(0,1)
  double lli, ll; // ini loglik and loglik in 2nd step of gibb
  double phi, lb, ub; // 3rd step of gibbs, draw from U(0,2pi), lower bound and upper bound
  vec Delta(2*p-1), DeltaPri(2*p-1); // 4th step of gibbs
  
  // OLS estimator: alpha_hat, beta_hat, gamma_hat, betastar_hat
  XZ = trans(X) * Z;
  XX = trans(X) * X;
  WW = trans(W) * W;
  WY = trans(W) * Y;
  betastar_hat = inv(WW) * WY;
  alpha_hat = betastar_hat(0);
  beta_hat = betastar_hat.tail(p-1);
  gamma_hat = inv(XX) * XZ;
  
  // a initial value of the derivation from the mean
  alpha = 1.1 * alpha_hat;
  beta = 1.1 * beta_hat;
  gamma = 1.1 * gamma_hat;
  betaC = gamma;
  betaD = alpha * gamma + beta;
  // a initial value of Delta
  Delta(0) = alpha - alpha_hat;
  Delta.subvec(1, p-1) = beta - beta_hat;
  Delta.subvec(p, 2*p-2) = gamma - gamma_hat;
  
  // loop, stop until get enough posterior samples
  while (h < nsamps){
    // std::cout<<h<<std::endl;
    // 1st step; random generate zeta from normal distribution with mean 0 variance sigma^2*inv(t(X)X)
    zeta.subvec(0, p - 1) = (randn(1, p) * chol(sigma2_nu*inv(WW))).t();
    zeta.subvec(p, 2*p - 2) = (randn(1, p-1) * chol(sigma2_ep*inv(XX))).t();
    
    // 2nd step; nu~U(0,1)
    nu = arma::as_scalar(randu(1));
    // initialize log likelihood value
    if (priorType == 1){
      lli = LogHorseshoePrior(betaC, HyperShr) + LogHorseshoePrior(betaD, HyperShr) + log(nu);
    } else if (priorType == 2){
      lli = log_stdnormal(betaC) + log_stdnormal(betaD) + log(nu);
    }
    
    // 3nd step; Draw angle from phi~U(0,2*pi)
    phi = arma::as_scalar(randu(1)*2*M_PI);
    lb = phi - 2.0 * M_PI;
    ub = phi;
    
    // 4th step;
    DeltaPri = Delta * cos(phi) + zeta * sin(phi);
    alphaPri = alpha_hat + DeltaPri(0);
    betaPri = beta_hat + DeltaPri.subvec(1, p-1);
    gammaPri = gamma_hat + DeltaPri.subvec(p, 2*p-2);
    betaCPri = gammaPri;
    betaDPri = alphaPri * gammaPri + betaPri;
    
    // 5th step;
    int iter = 0;
    if (priorType == 1){
      ll = LogHorseshoePrior(betaCPri, HyperShr)
               + LogHorseshoePrior(betaDPri, HyperShr);
    } else if (priorType == 2){
      ll = log_stdnormal(betaCPri) + log_stdnormal(betaDPri);
    }
    while (ll < lli){
      // std::cout<<h<<","<<iter<<std::endl;
      // 5a
      if (phi < 0){
        lb = phi;
      }else{
        ub = phi;
      }
      // 5b
      phi = runif(1, lb, ub)[0];
      // 5c
      DeltaPri = Delta * cos(phi) + zeta * sin(phi);
      alphaPri = alpha_hat + DeltaPri(0);
      betaPri = beta_hat + DeltaPri.subvec(1, p-1);
      gammaPri = gamma_hat + DeltaPri.subvec(p, 2*p-2);
      betaCPri = gammaPri;
      betaDPri = alphaPri * gammaPri + betaPri;
      
      if (priorType == 1){
        ll = LogHorseshoePrior(betaCPri, HyperShr)
        + LogHorseshoePrior(betaDPri, HyperShr);
      } else if (priorType == 2){
        ll = log_stdnormal(betaCPri) + log_stdnormal(betaDPri);
      }
      
      iter++;
    }//end prior eval
    
    // 6th step;
    Delta = DeltaPri;
    alpha = alpha_hat + DeltaPri(0);
    beta = beta_hat + DeltaPri.subvec(1, p-1);
    gamma = gamma_hat + DeltaPri.subvec(p, 2*p-2);
    betaC = gamma;
    betaD = alpha * gamma + beta;
    
    // final step; additional step: sample alpha
    mat Yw = Y - X * betaD;
    mat Zw = Z - X * betaC;
    if (finalStep == 2){
      double ZwZw_inv = arma::as_scalar(inv(trans(Zw) * Zw));
      double ZwYw = arma::as_scalar(trans(Zw) * Yw);
      alpha = as<double>(rnorm(1, ZwZw_inv * ZwYw, sigma2_nu * ZwZw_inv));
    } else if (finalStep == 1){
      alpha = arma::as_scalar(inv(trans(Zw) * Zw) * trans(Zw) * Yw);
    }

    // output results
    asamps(h) = alpha; // posterior samples alpha
    bcsamps.col(h) = betaC; // posterior samples betaC
    bdsamps.col(h) = betaD; // posterior samples betaD
    
    // update the global shrinkage parameter
    double HyperShr_tmp = exp(log(HyperShr) + as_scalar(randn(1))*0.05);
    double ratio = exp(LogHorseshoePrior(beta.tail(p-1), sigma2_nu * HyperShr_tmp)+
                       log_normal_density(HyperShr_tmp, 0.0, 100.0) - 
                       LogHorseshoePrior(beta.tail(p-1), sigma2_nu * HyperShr)-
                       log_normal_density(HyperShr, 0.0, 100.0)+
                       log(HyperShr_tmp) - log(HyperShr));
    if(as_scalar(randu(1)) < ratio){
      HyperShr = HyperShr_tmp;
    }
    
    
    // 
    h++; // update h
  }//end while(h<nsamps)
  
  amean = mean(asamps.subvec( round(nsamps * 0.1),nsamps - 1)); // posterior mean of alpha
  asd = stddev(asamps.subvec( round(nsamps * 0.1),nsamps - 1)); // posterior sd of alpha
  // for (int i = 0; i < p; i++){
  //   bcmeans(i) = mean(bcsamps.cols(500,999).row(i)); // posterior mean
  //   bcsds(i) = stddev(bcsamps.cols(500,999).row(i)); // posterior standard deviation
  //   bdmeans(i) = mean(bdsamps.cols(500,999).row(i)); // posterior mean
  //   bdsds(i) = stddev(bdsamps.cols(500,999).row(i)); // posterior standard deviation
  // }
  return List::create(_["alpha"] = amean, _["alpha_SD"] = asd, _["asamps"] = asamps);
}//end BayesLMTEE2M_horse()












