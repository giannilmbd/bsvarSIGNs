
# remove.packages('bsvarSIGNs')
remotes::install_github("giannilmbd/bsvarSIGNs",force=F)

# install.packages('bsvarSIGNs')
library(bsvarSIGNs)
# investigate the effects of the optimism shock
data(optimism)

# specify identifying restrictions:
# + no effect on productivity (zero restriction)
# + positive effect on stock prices (positive sign restriction) 
sign_irf       = matrix(c(0, 1, rep(NA, 23)), 5, 5)

# specify the model and set seed
set.seed(123)
specification  = specify_bsvarSIGN$new(optimism * 100,
                                       p        = 12,
                                       sign_irf = sign_irf)

# estimate the model

posterior      = estimate(specification, S = 200,n_draws = 30,draw_strategy=3)#


irf            = bsvars::compute_impulse_responses(posterior, horizon = 40)

plot(irf)
