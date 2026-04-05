# Assignment 17.1: Comparing Classifiers

## Overview

The goal of this assignment is to compare the performance of four classifiers:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)

You will apply these classifiers to a real-world dataset from a Portuguese banking institution, which contains the results of multiple telephone-based marketing campaigns for bank term deposit subscriptions.

---

## Data

The dataset comes from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). It captures client demographics, campaign contact details, and macroeconomic indicators, along with whether each client ultimately subscribed to a term deposit.

Two formats of the dataset are available. This assignment will use the **20-feature format** (`bank-additional-full.csv`) as the primary dataset, with the 15-feature format (`bank-full.csv`) used for comparison time permitting.

Run `python src/setup.py` from the project root to download all data files into the `data/` directory.

### Dataset: 20-Feature Format (`bank-additional-full.csv`)

This is the richer of the two formats. It adds five macroeconomic context indicators not present in the 15-feature version. There are **41,188 rows** and **21 columns** (20 features + 1 target).

#### Column Definitions

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `age` | numeric | Client's age in years. |
| 2 | `job` | categorical | Client's occupation (e.g., `admin.`, `blue-collar`, `technician`, `retired`, `student`, `unemployed`, `unknown`). |
| 3 | `marital` | categorical | Marital status: `married`, `single`, `divorced`. (`divorced` includes widowed.) |
| 4 | `education` | categorical | Highest education level: `illiterate`, `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `professional.course`, `university.degree`, `unknown`. |
| 5 | `default` | binary | Whether the client has credit currently in default: `yes`, `no`, `unknown`. A client in default has missed loan payments. |
| 6 | `housing` | binary | Whether the client has an active housing (mortgage) loan: `yes`, `no`, `unknown`. |
| 7 | `loan` | binary | Whether the client has an active personal loan: `yes`, `no`, `unknown`. |
| 8 | `contact` | categorical | How the client was reached: `telephone` (landline) or `cellular` (mobile). |
| 9 | `month` | categorical | Month of the last contact in the current campaign (`jan`–`dec`). |
| 10 | `day_of_week` | categorical | Day of the week of the last contact (`mon`–`fri`). |
| 11 | `duration` | numeric | Duration of the last call in seconds. **Note:** this column is highly predictive but is only known after the call ends. It should be excluded from realistic predictive models but is useful for benchmarking. |
| 12 | `campaign` | numeric | Total number of times this client was contacted during the current campaign (including the last contact). |
| 13 | `pdays` | numeric | Number of days since the client was last contacted in a *previous* campaign. `999` means the client was never previously contacted. |
| 14 | `previous` | numeric | Number of times the client was contacted before this campaign. |
| 15 | `poutcome` | categorical | Outcome of the previous marketing campaign for this client: `success`, `failure`, `nonexistent`. |
| 16 | `emp.var.rate` | numeric | Employment variation rate — a quarterly economic indicator reflecting changes in employment levels. Negative values indicate job losses; positive values indicate growth. |
| 17 | `cons.price.idx` | numeric | Consumer Price Index (CPI) — a monthly indicator of inflation based on the price of a basket of consumer goods. Higher values indicate higher inflation. |
| 18 | `cons.conf.idx` | numeric | Consumer Confidence Index — a monthly survey-based indicator of how optimistic consumers feel about the economy. More negative values reflect lower confidence. |
| 19 | `euribor3m` | numeric | Euribor 3-month rate — the daily interest rate at which European banks lend to each other for 3-month terms. A proxy for the prevailing interest rate environment. |
| 20 | `nr.employed` | numeric | Number of employees in the economy (in thousands) — a quarterly indicator of overall labor market size. |
| 21 | `y` | **target** | Whether the client subscribed to a term deposit: `yes` or `no`. This is the outcome variable you are trying to predict. |

> **Macroeconomic columns (16–20)** are the key additions in this format. They capture the broader economic climate at the time of each call, providing context that individual client attributes alone cannot capture.

---

## Deliverables

Build a **Jupyter Notebook** that includes:

1. A clear statement of the business problem and why it matters
2. Exploratory data analysis with appropriate visualizations
3. Data cleaning and preparation steps
4. Training and comparison of all four classifiers
5. Correct interpretation of descriptive and inferential statistics
6. A findings section with actionable insights written for a non-technical audience
7. Next steps and recommendations

---


