# TU Berlin - Speech and Audio Technology in Medicine Seminar - WS24

# Independent Component Analysis (ICA)

- [TU Berlin - Speech and Audio Technology in Medicine Seminar - WS24](#tu-berlin---speech-and-audio-technology-in-medicine-seminar---ws24)
- [Independent Component Analysis (ICA)](#independent-component-analysis-ica)
  - [Overview](#overview)
  - [Links](#links)
  - [Demo](#demo)
    - [How to install](#how-to-install)
    - [How to run](#how-to-run)

## Overview

This is a seminar presentation about the ICA procedure. ICA deals with receiving two signals which are mixed signals from two original source and estimating those original signals from the mixed signals.

## Links

- [Paper](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf)
- [MATLAB Code](http://research.ics.aalto.fi/ica/fastica/)
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
- [rust](https://docs.rs/linfa-ica/latest/linfa_ica/)

## Demo

This demo shows the workings of the FastICA algorithm using `linfa-rs`, a Rust implementation very similar to Python's scikit-learn for Machine Learning tasks.

### How to install

First, you will need to install Rust itself. Go to [rustup.rs](https://rustup.rs/) and follow the instructions on screen. This will install everything you need to run this demo (If you are running Ubuntu or Fedora Linux, you will also need to install some [dependencies](https://github.com/plotters-rs/plotters?tab=readme-ov-file#dependencies) to be able to plot the output).

### How to run

With Rust installed you are now ready to run the demo, open a terminal in the [/demo](/demo/) subdirectory and run `cargo`.

```bash
cd demo
# set the release flag for slower compile time but much faster execution time
cargo run [--release]
```