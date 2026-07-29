'mean': the mean of the previous n values.
'std': the standard deviation of the previous n values.
'min': the minimum of the previous n values.
'max': the maximum of the previous n values.
'sum': the sum of the previous n values.
'median': the median of the previous n values.
'ratio_min_max': the ratio between the minimum and maximum of the previous n values.
'coef_variation': the coefficient of variation of the previous n values.
'ewm': the exponentially weighted mean of the previous n values. The decay factor alpha can be set in the kwargs_stats argument, default is {'ewm': {'alpha': 0.3}}.

- Muy importante conocer el proceso físico y químico para tener información extra de la reacción de las variables
    - Variables de control, sensores, variable objetivo