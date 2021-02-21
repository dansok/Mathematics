Every problem has a dedicated file and can be run as a regular python program.
All the parameters are hard coded. requirements can be fuond in stochastic_analysis/requirements.txt


### Results ###

(1)
Exponential Random Variable with mean 1 has the following output in the terminal --

```
Question #1: Monte Carlo Simulated mean of exponential RVs with mean 1: 0.9984592053547269
```


(2)
The resulting plot for the weak law of large numbers is under plots/
we also show a theoretical upper bound due to Chebyshev's inequality


(3)
The resulting plot for the central limit theorem is under plots/


(4)
The resulting plot for the large deviations principle / concentration inequality in under plots/
We show that 1/N * the log probabilities -> -constant(epsilon). Note that the constant is negative,
since log(0,1) = (-\infty, 0)
