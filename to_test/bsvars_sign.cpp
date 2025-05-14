// Modified bsvar_sign_cpp with draw_strategy input and restored correct OpenMP parallel block

#include <RcppArmadillo.h>
#include "progress.hpp"
#include "Rcpp/Rmath.h"
#include <bsvars.h>
#include "sample_hyper.h"
#include "sample_Q.h"
#include "sample_NIW.h"
#include <mutex>
#include <random>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::interfaces(cpp)]]
// [[Rcpp::export]]
Rcpp::List bsvar_sign_cpp(
    const int&        S,
    const int&        p,
    const arma::mat&  Y,
    const arma::mat&  X,
    const arma::cube& sign_irf,
    const arma::mat&  sign_narrative,
    const arma::mat&  sign_B,
    const arma::field<arma::mat>& Z,
    const Rcpp::List& prior,
    const bool        show_progress = true,
    const int         thin = 100,
    const int&        max_tries = 10000,
    const int&        n_draws = 10,
    const int&        draw_strategy = 1  // 1 = overrep, 2 = prune, 3 = IS
) {
  
  std::string oo = (thin != 1) ? bsvars::ordinal(thin) + " " : "";
  
  Progress bar(50, show_progress);
  const int  T = Y.n_rows, N = Y.n_cols, K = X.n_cols;
  const int total_draws = S * n_draws;
  
  vec  posterior_w(total_draws);
  mat  posterior_hyper(N + 3, total_draws);
  cube posterior_A(N, K, total_draws);
  cube posterior_B(N, N, total_draws);
  cube posterior_Q(N, N, total_draws);
  cube posterior_Sigma(N, N, total_draws);
  cube posterior_Theta0(N, N, total_draws);
  cube posterior_shocks(N, T, total_draws);
  
  mat hypers = as<mat>(prior["hyper"]);
  int S_hyper = hypers.n_cols - 1;
  int prior_nu = as<int>(prior["nu"]);
  int post_nu = prior_nu + T;
  
  mat prior_B = as<mat>(prior["B"]);
  mat Ysoc = as<mat>(prior["Ysoc"]);
  mat Xsoc = as<mat>(prior["Xsoc"]);
  mat Ysur = as<mat>(prior["Ysur"]);
  mat Xsur = as<mat>(prior["Xsur"]);
  vec prior_v = as<mat>(prior["V"]).diag();
  
  std::mutex write_mutex;
  int total_valid_draws = 0;
  int s = 0;
  
  while (s < S) {
    vec hyper = hypers.col(randi(distr_param(0, S_hyper)));
    double mu = hyper(0), delta = hyper(1), lambda = hyper(2);
    vec psi = hyper.rows(3, N + 2);
    
    mat prior_V = diagmat(prior_v % join_vert(lambda * lambda * repmat(1 / psi, p, 1), ones<vec>(K - N * p)));
    mat prior_S = diagmat(psi);
    
    mat Ystar = join_vert(Ysoc / mu, Ysur / delta);
    mat Xstar = join_vert(Xsoc / mu, Xsur / delta);
    mat Yplus = join_vert(Ystar, Y);
    mat Xplus = join_vert(Xstar, X);
    
    field<mat> result = niw_cpp(Yplus, Xplus, prior_B, prior_V, prior_S, prior_nu);
    mat post_B = result(0), post_V = result(1), post_S = result(2);
    post_nu = as_scalar(result(3));
    
    mat Sigma = iwishrnd(post_S, post_nu);
    mat chol_Sigma = chol(Sigma, "lower");
    mat B = rmatnorm_cpp(post_B, post_V, Sigma);
    mat h_invp = inv(trimatl(chol_Sigma));
    
    cube Q_all(N, N, n_draws);
    cube shocks_all(N, T, n_draws);
    vec weights(n_draws, fill::zeros);
    int valid_draws = 0;
    
#pragma omp parallel shared(Q_all, shocks_all, weights, valid_draws, p, Y, X, prior, sign_irf, sign_narrative, sign_B, Z, max_tries) \
    firstprivate(Sigma, chol_Sigma, B, h_invp)
    {
      int local_valid_draws = 0;
#pragma omp for
      for (int i = 0; i < n_draws; i++) {
        int local_tries = 0;
        double local_w = 0;
        mat local_Q, local_shocks;
        
        while (local_w == 0 && (local_tries < max_tries || max_tries == 0)) {
          field<mat> local_result = sample_Q(p, Y, X, B, h_invp, chol_Sigma, prior, sign_irf, sign_narrative, sign_B, Z, 1);
          local_Q = local_result(0);
          local_shocks = local_result(1);
          local_w = as_scalar(local_result(2));
          local_tries++;
        }
        
        if (local_w > 0) {
#pragma omp critical
{
  Q_all.slice(valid_draws) = local_Q;
  shocks_all.slice(valid_draws) = local_shocks;
  weights(valid_draws) = local_w;
  valid_draws++;
}
        }
      }
    }
    
    if (valid_draws == 0) continue;
    
    std::lock_guard<std::mutex> lock(write_mutex);
    
    if (draw_strategy == 1) {
      for (int i = 0; i < valid_draws; i++) {
        int index = total_valid_draws++;
        posterior_w(index) = weights(i);
        posterior_A.slice(index) = B.t();
        posterior_B.slice(index) = Q_all.slice(i).t() * h_invp;
        posterior_Q.slice(index) = Q_all.slice(i);
        posterior_Sigma.slice(index) = Sigma;
        posterior_Theta0.slice(index) = chol_Sigma * Q_all.slice(i);
        posterior_shocks.slice(index) = shocks_all.slice(i);
      }
    } else if (draw_strategy == 2) {
      int j = rand() % valid_draws;
      int index = total_valid_draws++;
      posterior_w(index) = 1.0;
      posterior_A.slice(index) = B.t();
      posterior_B.slice(index) = Q_all.slice(j).t() * h_invp;
      posterior_Q.slice(index) = Q_all.slice(j);
      posterior_Sigma.slice(index) = Sigma;
      posterior_Theta0.slice(index) = chol_Sigma * Q_all.slice(j);
      posterior_shocks.slice(index) = shocks_all.slice(j);
    } else if (draw_strategy == 3) {
      double w = 1.0 / valid_draws;
      for (int i = 0; i < valid_draws; i++) {
        int index = total_valid_draws++;
        posterior_w(index) = w;
        posterior_A.slice(index) = B.t();
        posterior_B.slice(index) = Q_all.slice(i).t() * h_invp;
        posterior_Q.slice(index) = Q_all.slice(i);
        posterior_Sigma.slice(index) = Sigma;
        posterior_Theta0.slice(index) = chol_Sigma * Q_all.slice(i);
        posterior_shocks.slice(index) = shocks_all.slice(i);
      }
    }
    
    if (show_progress && s % (S / 50) == 0) bar.increment();
    s++;
  }
  
  posterior_w.resize(total_valid_draws);
  posterior_A.resize(N, K, total_valid_draws);
  posterior_B.resize(N, N, total_valid_draws);
  posterior_Q.resize(N, N, total_valid_draws);
  posterior_Sigma.resize(N, N, total_valid_draws);
  posterior_Theta0.resize(N, N, total_valid_draws);
  posterior_shocks.resize(N, T, total_valid_draws);
  
  if (draw_strategy == 3) {
    posterior_w /= sum(posterior_w);
  }
  
  return List::create(
    _["posterior"] = List::create(
      _["w"]        = posterior_w,
      _["hyper"]    = posterior_hyper,
      _["A"]        = posterior_A,
      _["B"]        = posterior_B,
      _["Q"]        = posterior_Q,
      _["Sigma"]    = posterior_Sigma,
      _["Theta0"]   = posterior_Theta0,
      _["shocks"]   = posterior_shocks
    )
  );
} // END bsvar_sign_cpp