# Questions

# Meeting 1 - 30.10.24

- moses registration until 30.11.24?
  - yes
- use rust (linfa_ica) instead?
  - yes, but give install instructions
- code demo, example of using the API and its output?
  - yes, show main fn and also make an example with real audio signals (plots and play them)
- what is $E = \{ ... \}?$
  - expected value of random variable
  - e.g. $E = \{y^3\}$: y is a random distribution of variables (for example a signal) and when averaged it has the expected value of $y^3$

# Meeting 2 - 20.11.24

- mp3 sample voice lines? is there a sample database or record a snippet?
  - for testing record a sample
  - there should be plenty of examples for specific use-cases
    - however, they might not work with FastICA (missed the name some other ICA method)
    - something about convoluted signal and different spectrums
- how so safe mixed signals? added in one file or separate?
  - separate files, one for each mixed signal, original signal and extracted signals (six total)
- references on slides, what if it is mostly the same?


condition matrix, for inversing matrix

# Meeting 3 - 11.12.24

- briefly list "Minimization of Mutual Information", "Maximum Likelihood Estimation" and "The Infomax Principle" as alternatives to Approximation of Negentropy?
  - not necessary! they are just different ways to get to the same
- Kurtosis, Negentropy, Approx
  - only do this if there is time for it
  - Kurtosis, peekines of signals, could be shown with plots, gaussian, laplace, subgaussian, etc.
- formulas?
  -  the important ones for sure, mixed signals, uncorrelatation, whitening
  -  the others only if I can explain them really well, otherwise just outline what is supposed to happen
-  demo
   -  record two signals and mix them
   -  play them, original, mixed, unmixed
   -  explain FastICA API
   -  show signal plots, recordings and simple (sine + box signal)