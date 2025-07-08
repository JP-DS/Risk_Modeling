# Term Project Instructions

#### Sept. 23 , 2024

### Assignment

To help you gain experience working with real-world data sets, this project will require that you mine actual bank transaction data
to solve a problem of interest to your client, a large Italian bank seeking to reduce its losses from bad loans. You will be working in
teams of 2 - 4 students. Please ensure that all students participate equally.

This project will require that you mine actual transaction data to solve a problem of interest to your client, Banca Massiccia. a large
Italian bank that is interested in understanding how to reduce its portfolio default rate on individual loans.

#### In particular, the firm wishes to get better predictions of the one-year probability of default (PD) for a prospective borrower.^1

### Context Overview

Imagine that you have been approached by Banca Massiccia, a large Italian bank. Banca Massiccia has been making loans to
businesses for many years, and has experimented with different approaches to underwriting these loans, including statistical models
of default. The Head of Loan Origination has asked you to assist the firm in optimizing its underwriting using the power of machine
learning. The goal is to produce estimates of the probability of default (PD) for prospective borrowers so the bank can use risk-
based pricing to set interest rates and underwriting fees for borrowers. One way to do this is to leverage the data that Banca
Massiccia has accumulated over the past several years.

#### The basics of the transaction flow are as follows:

1. A potential borrower applies for a loan by providing information about the finances of the firm in the form of financial
    statement data.
2. The loan officer assigned to the borrower analyzes the financial data using a number of human and automated approaches
    to try to determine how likely it is that the borrower will default.
3. The bank officer then determines what the appropriate interest rate and underwriting fees would be for the loan to be
    profitable, in consideration of the default probability.
4. If the bank officer concludes that the loan may be made at a rate that is within the bank’s guidelines, the loan is made,
    and the customer’s information is transferred to the loan monitoring group, which continues to monitor both changes in
    the financial status of the borrower, and the payment status of the loan.
5. If the borrower defaults before the loan is repaid, the borrower’s case is transferred to the “work-out” group to try to
    recover the shortfall through various legal remedies.

### Specific Business need: Predicting Probability of Default

Banca Massiccia would like to be able to better predict the probability that a potential borrower will default on a principal or
interest payment for a prospective loan over the next 12 months.

The bank has asked you to design the data mining task, mine the data, and describe your results. The management is asking you to
design the data mining task, mine the data, and describe your results. You will also need to research existing solutions to the
problem. At the end of the semester we will conduct a tournament to evaluate the relative performance of your model as well.

For purposes of the Term Project, you can think of your stakeholder(s) as internal members of the bank’s lending group and risk
management group (in other contexts, the stakeholder might be a consulting client or a VC/incubator who may fund you). You will

(^1) This data and problem are from real-world Italian companies. The data is real-world data. However, some of the details of the data have been modified for
confidentiality.

## DS – GA 3001


need both to inform the stakeholder about your project and results and provide context by reporting on what has been done to
date (elsewhere) on the problem.

It is most important to develop your models and analysis study well (within the scope of what we’ve discussed in class) and
with a good understanding of the business problem.

You should use the frameworks and methodologies we used this semester to structure your project and writeup. Keep in mind that
it may be ineffective simply to proceed linearly through the steps, and this may need to be reflected in your analysis. You can
interact with me and your TA from the preparation of your initial ideas through your write-up, as a consulting group would interact
with a stakeholder or funding source in preparing a research report. However, given the class size, and depending on the number of
groups, there may not be as many opportunities to do so as you (or I) would like. Use your imagination, prior experience, or ask us
to help fill in any gaps in your understanding.

### Deliverables

#1: By Sept 2^3 you will submit your choices for teams for projects to your TA. You should self-organize your teams, but ask us if
you need help. Please include a paragraph or two about what you are thinking you might do. There is also a section of Ed
Discussion called “Students seeking teams” that we set up to help you find teams and/or members.

#2: By Oct 5 th you will submit a proposal for your project with as much detail as possible your ideas. This is where you describe
your detailed ideas on how you will formulate the problem. Review your notes on problem formulation before you start this.
Be sure to include: the exact (business) problem; the use scenario, the related data mining problem (and whether it is
supervised or unsupervised; the unit of analysis; potential target variables (if supervised); potential features; how the results
solve the business problem? etc. As well as why you are approaching this finance problem through your formulation and what
finance characteristics make your formulation valuable. Similarly, please describe how you will formulate that validation of
your model. Feedback will be provided by Roger during office hours which each team will schedule independently.

# 3 : By Oct 27 th^ you will submit a status report (“Project Update”) including preliminary & results, validation, hypotheses, and any
issues that you have run into.

# 4 : By Nov 16 th, you will submit your final write-up and code.

#### Your write-up:

1. should include the information detailed in the next section, in approximately the order given.
2. be about 15 – 25, fairly dense (but readable) slides, including any appendices you would like to include. Provide clear
    citations and bibliography for the external sources you use. More detail is given below.
3. An appendix detailing the contributions of each team member. All group members should contribute to the analysis
    and write-up is also required

#### Your code will include three main modules:

1. The code you used to estimate your solution (with in-line comments and documentation where necessary);
2. The code that you want us to use to make new predictions using your model, but which does not use the
    holdout data to estimate any parameters; and
3. A “harness” function that will allow us to upload new data into your fitted solution (in 2, above) by calling a
    single function. Your harness should produce a list of results that we can use for validation.

Details on these components are given in the Appendix A.

Details of the data that you will be using are given in Appendix B.


## Appendix A: Project Deliverables and Evaluation

#### Holdout-sample evaluation

In addition to reviewing your writeup, I have reserved a portion of the data in a vault at an undisclosed location. We will use
this data set to evaluate your true out-of-sample performance using the harness code that you provide.

You may assume that the record structure of the holdout sample will be the same as that of the full training data sample. Be
aware that there may be borrowers in the hold-out sample that may not have been in the training data.

Your model will be evaluated in terms of the accuracy of the probabilities it produces and in terms of its discriminatory power. We
will also examine model robustness and face validity (i.e., do the variables and parameters of the model make sense). For purposes
of the horserace, we will be examining the pairwise statistical and practical differences in the metrics we calculate, i.e., examining all
possible tournaments of model pairs for performance differences using the evaluation criteria we will learn over the semester.

Output specifications:

- that your model should produce an estimate of the probability of default (PD) for each new record, i.e. one, and only one,

#### probability for each record. You must return these predictions in a n x 1 vector, where n is the number of rows in the hold-

```
out sample. YOU MUST PRODUCE A PROBABILITY FOR EVERY RECORD IN THE HOLDOUT DATA. (Note that 1/0 is
technically a PD.)
```
- Your model should not depend in any way on the hold-out sample
- Remember to only use data that you would have access to in real life.
- Remember not to use label vars (e.g., ‘HQ_city’) as numerical values.
- Please (please, please!) look at what your model is doing in detail before submitting it.

#### Pitch Deck

Your pitch-deck should be something that you would use to explain your work to a client/colleague and should address both the
positive aspects of your solution and its quantitative and economic limitations, etc. Your pitch book can be any length you like
(typical submissions contain 15-25 slides with fairly dense, but readable, information content), and may include one or more
technical appendices you think are helpful.

You may wish to include the information detailed below. Your write-up need not have corresponding sections or bullet points, but

#### we should be able to find the information without searching too hard. Be as precise/specific as you can.

Business Understanding (take this seriously)

- Identify, define, and motivate the business problem.
- How (precisely) will a data mining solution address the business problem?
- What has been done in the past both from a finance and a machine learning perspective?

```
Example of how to think about reviewing previous work for stakeholders: For this project, we are working on loan data from a
large Italian bank. One way to think about how to present previous work, is to imagine that you are pitching your idea to a
principal a VC firm that has experience funding a large number of machine-learning-based FinTech startups. The VC wants you to
explain how your approach is different from what the firm already knows.
```
```
You would want to discuss:
```
- any good ideas that you incorporated from earlier work,
- how your default probability estimation approach is different from current approaches, and
- why your approach is better (uniformly or in specific settings only) for loan underwriting.

```
Don’t fall into the trap of only reviewing machine learning research that has been done on this problem. Much of it is not really
informed by finance. You will benefit by getting familiar with both the finance and machine learning literature on this problem.
```

```
Problem formulation including what has been done in the past and why your approach is better (uniformly or in specific
settings only):
```
- Did you set this problem up to make it easier for your algorithm to learn the problem?
- How did you incorporate financial intuition and theory into your model?
- Why will this formulation be more successful than if you had just mined the data without a finance context?

```
Data Understanding including which data you used and what was problematic or particularly useful about it, as well as biases
& limitations.
```
- Identify and describe the data that you used to address the business problem. Include those aspects of the data that we routinely talk
    about in class and/or in the homeworks, including variable definitions, calculated or derived variables, and, of course, potential biases.

```
Data Preparation including how you have filtered, cleaned, preprocessed and munged the data. Discuss limitations of your data
handling.
```
- Specify how these data are integrated to produce the format required for data mining.
- Describe any preprocessing you did on individual features or variables.
    (NB: data preparation can be time consuming! Get started early.)

```
Modeling including a description of the approaches you explored and the formulations you used in setting up the problem to
let the algorithms you used learn efficiently in a finance-aware context. Discuss which variables you selected (and why/how).
This is also a good place to discuss the model itself and any insights the team got from diagnosing and examining the model(s)
parameters and variables.
```
- Specify the type of model(s) built and/or patterns mined.
- Discuss choices for data mining algorithm: what are alternatives, and what are the pros and cons?
- Describe the final set of variables used in your final model.
- Discuss the economic intuition for your model and results.
- Discuss why & how this model should “solve” the business problem (i.e., improve along some dimension of interest to the firm).
- Make sure to actually include a full specification of your final model!

```
Evaluation including how you set up tests and evaluated your solution, and how it compared to other approaches, etc.. Include
the specific results of your evaluation, benchmarks, etc.
```
- Discuss how the result of the data mining is/should be evaluated and how you evaluated your model.
- Discuss any benchmarks you used to evaluate relative performance and provide performance analysis.
- Suggest how a business case should be developed to project expected improvement? ROI?
- If this is impossible/very difficult, explain why and identify any viable alternatives.

```
Deployment
```
- Discuss how the result of the data mining will be deployed
- Discuss issues the bank should be aware of regarding deployment.
    o Are there important ethical or legal considerations?
    o Describe settings in which your results should not be applied.
    o Identify the risks associated with your proposed plan and how you would mitigate them.

You will probably find a number of useful sources in the library as well. These can be invaluable. The first individual to notice this
line and contact me will be rewarded for the effort. Please make sure to cite any literature and sources you used in your work, and
provide a bibliography referencing the source.

All team members should contribute to the analysis & write-up. Make sure to include an appendix that describes each team
member’s contribution to the work. Remember, the team may be asked to actually pitch your solution to the class. Make sure you
have what you need inside the pitch book to do so.

Where useful, include model output, data visualizations, etc. to explain what you did and found, the structure of your model, etc.

Include a detailed representation of your final model (e.g., regression output, a tree, etc.), & be prepared to discuss the economic
intuition for it.


#### Model codebase

Your codebase will include three main modules implemented in R or Python:^2

1. Your estimation code: The code you used to estimate your solution (with in-line comments and documentation where
    necessary);
2. Your prediction code: The code that you want us to use to make new predictions using your model, but which does not use
    the holdout data to estimate any parameters; and
3. Your harness code: A “harness” function that will allow us to upload new data into your fitted solution (in 2, above) by
    calling a single function. Your harness should produce a list of results that we can use for validation.

Your estimation code may use any algorithms you have studied, except for LLMs or other large neural networks. Feed-forward
ANNs are permitted.

Your prediction code may take in one or more rows of data, and produce output in a form that you find useful, provided it does
not use the test data for any purpose (e.g., you cannot calculate the mean of a variable in the test data to scale that variable new
record). You may perform any legitimate data transformation and normalization before running your model, and you may post-
process your model output using any legitimate operations you prefer.

Your harness code should run with a single command in our environment on reasonably configured hardware without throwing
any errors or halting unexpectedly. It should not and not require any knowledge of your model or code to run properly.

Make sure that any parameters you need to use (e.g., mean and SD for standardization) are either hardcoded, or in a parameter file
that you provide. If you use the training data dynamically (e.g., for KNN), please make sure to provide a data file in the form that
you need for your harness.

Your harness should not re-estimate your model at run time. Make sure that any preprocessing logic (e.g., transformations,
missing value treatment, creation of dummies, etc.) is done automatically by the harness. Note that if you wish to use external
data fields that are not in the training/holdout data, please:

- Check with us at least several weeks before the due date to make sure your use of these variables is practical and
    compliant.
- make sure that your harness fetches and handles these properly without peeking into the future, and without the need for
    us to do anything special.

The code for the harness and the various components of the model, etc. should be well documented.

We will not debug your code if it does not run. Make sure your model runs in your harness! Although it is not required, you may
submit a test version of your harness and model to the TA one week before the project is due so that you can determine whether it
will run properly on the data we will be using. Any model that cannot run on our data without modification will be disqualified
from the horserace and will not be eligible for the bonus point prize. Even if your model does not run, we will still evaluate and
grade the other aspects of your project.

### Presentation:

Depending on the number of teams, the top _k_ projects will be selected and the project teams will pitch their solution to the class.

#### Note that in most cases, k<n , where k and n are the number of teams presenting and total number of teams, respectively.

- You will get the most out of the project if you interact with us during the development of your ideas.
- And please feel free to come talk to us about your ideas as time permits.

(^2) You may use R or Python to conduct your research, EDA, etc. However, if you plan to submit your final code in R, please let us know at the time you submit your
proposal. If you do not, only Python code will be accepted.


## Appendix B: Data description

### Data Overview

The IT department at Banca Massiccia has agreed to provide you with information on previous borrowers, including information
about the company type, industry, etc. as well as financial statement data on an annual basis for each borrower. Because the firm is
located in the EU, where privacy and data sharing are heavily regulated, some of the data that Banca Massiccia collects may not be
shared with your team. The Data Dictionary section, below, provides the definitions of the variables in the training data set.

### Data Conventions

#### The data in both the training and holdout samples will have identical structures and will conform with the following

#### conventions:

- Each row is one firm-year
- Annual observations
- 44 variables in total
- All quantities (except ratios) are reported n €
- Only firms with > €1.5MM in assets are included in data
- Only non-finance/insurance firms are included
- There is no information in the training data regarding defaults that occurred after 12/31/
- The value for the default date (def_date) is NA in the holdout sample

#### The holdout sample is drawn from a future time period and will include both firms that are in the training data and those that

#### are not.

#### The training and holdout samples will have identical record structures and will conform with the following conventions the

#### data dictionary given on the next page.


#### train.csv (financial statements and default dates behavior)

```
Variable name
(column name, feature, etc.) Description
id Firm identifier
HQ_city City of main branch
legal_struct Legal structure of the firm
ateco_sector Industry sector code (see ATECO sector definition doc)
```
**fs_year** (^) Year of the financial statement
**asst_intang_fixed** (^) Intangible assets
**asst_tang_fixed** Tangible Assets
**asst_fixed_fin** Financial assets
**asst_current** Current assets
**AR** Accounts receivable
**cash_and_equiv** Cash & equivalent holdings
**asst_tot** Total assets
**eqyt_tot** Total equity
**eqty_corp_family_tot** Total equity for entire group ("family")
**liab_lt** Long-term liabilities
**liab_lt_emp** (^) Long-term liab to employees
**debt_bank_st** (^) Short-term bank debt
**debt_bank_lt** Long-term bank debt
**debt_fin_st** Short-term debt other
**debt_fin_lt** Long-term debt other
**AP_st** Short-term accounts payable
**AP_lt** Long-term accounts payable
**debt_st** Short-term debt
**debt_lt** (^) Long-term debt other
**rev_operations** (^) Operating revenue
**COGS** (^) COGS (Cost of goods sold)
**prof_operating** (^) Operating profit
**goodwill** Goodwill
**inc_financing** Financial income
**exp_financing** Financial expenses
**prof_financing** Financial profit
**inc_extraord** Extraordinary income
**taxes** Taxes
**profit** (^) Net profit
**days_rec** (^) Days recievables
**ebitda** Earnings before interest, taxes, depreciation, and amortization
**roa** (^) Return on assets
**roe** Return on equity
**wc_net** Net working capital
**margin_fin** (Equity - Fixed assets)
**cf_operations** Operating cashflow


