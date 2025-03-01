# 2025-iaqf-mafn

Abstract— The S&P 500 has long served as a benchmark for overall stock market performance. However, its capitalization-weighted structure has led to an increasing concentration in a handful of mega-cap stocks, particularly the Magnificent Seven, which have disproportionately driven index returns in recent years.
This paper examines the implications of this concentration for using the S&P 500 as a representative market proxy within the investment process. We further analyze the consequences of such concentration on index options pricing and volatility from the perspective of a dispersion
trader. Lastly, we explore the challenges this dynamic poses for institutional equity long/short investors and propose potential adjustments to mitigate concentration risk while preserving desired market exposure.

This repository contains our team's code for the 2025 IAQF competition. The code is organized by team member.

## Main Files

The main files used for each question are highlighted below.

### Question 1

* `austin/download.ipynb`: data fetching.
* `bill/Question 1/`: This directory contains individual CSV files for each stock analyzed in Question 1, along with CSV files for the S\&P 500 and the industry index data.
* `thomas/q1/iaqf-q1-v1.ipynb`: This notebook contains analysis for section II of the paper, such as weighted index beta calculations and rolling CAPM regressions etc.
* `thomas/alex/competition.py`: Python script with functions used in the analysis.

### Question 2

* `austin/implied_corr_model.ipynb` for simulation of 2 or multi-asset setting, discussing the impact of concentration on implied correlation.
* `alex/iaqf_kurtosis.py`: the simulated kurtosis of portfolios with different concentrations.
* `paul/Dispersion Graphics.ipynb` for historical charts of basket vs implied volatility.

### Question 3

* `thomas/q3/iaqf-q3-v1.ipynb`: Jupyter Notebook for Question 3 analysis.
* `thomas/q3/iaqf-q3-v1.ipynb`: This notebook contains the backtest model utilized in section V of the paper. It also contains historical performance summaries of size factor returns, and factor risk decompositions.
* `alex/iaqfvol.py`: Python script related to volatility analysis.

*Note: Some files may be shared or used across different questions.