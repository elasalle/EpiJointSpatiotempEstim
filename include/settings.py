# # Fixed parameters ------------------------------------------------------------------------------------------------

# Phi parameters
phiAlpha = 1 / 0.28
phiBeta = 1.87
phiDays = 26

# Sliding median parameters
slidingMADDays = 14

# Prior parameters for Cori's method (C) only
priorA = 1
priorB = 5
tauWindow = 15

# Building synthetic data threshold
thresholdPoisson = 0.1


# Regularization parameters examples (lambdaR, lambdaO)
RegularizationSettings = {'Fast':       (3.5, 0.03),    # for fast trends with a lot of outliers
                          'Slow':       (50 , 0.75),    # for slow trends with few outliers
                          'lowLambdaO': (50 , 0.03),    # for slow trends with a lot of outliers
                          'lowLambdaR': (3.5, 0.75)}    # for fast trends with few outliers

# Used parameters for generation of 1D data with configurations I, II, III and IV.
Configs = {'I': {'name': 'Fast',
                 'country': 'France',
                 'fday': '2021-11-01',  # 276 days /!\ containing 3 epidemic waves
                 'lday': '2022-08-03',
                 'lambdaR': 3.5,
                 'lambdaO': 0.03,
                 'useCase': 'great variations and much outliers'},
           'II': {'name': 'Slow',
                  'country': 'France',
                  'fday': '2021-11-01',
                  'lday': '2022-08-03',
                  'lambdaR': 50,
                  'lambdaO': 0.03,
                  'useCase': 'few variations and few outliers'},
           'III': {'name': 'CrossedRFastOSlow',
                   'country': 'France',
                   'fday': '2021-11-01',
                   'lday': '2022-08-03',
                   'lambdaRR': 3.5,
                   'lambdaOR': 0.03,
                   'lambdaRO': 50,
                   'lambdaOO': 0.75,
                   'useCase': 'great variations and few outliers'},
           'IV': {'name': 'CrossedRSlowOFast',
                  'country': 'France',
                  'fday': '2021-11-01',
                  'lday': '2022-08-03',
                  'lambdaRR': 50,
                  'lambdaOR': 0.75,
                  'lambdaRO': 3.5,
                  'lambdaOO': 0.03,
                  'useCase': 'few variations and much outliers'},
           }
