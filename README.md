# reddit-topic-modelling

## Characteristics

Given JSON files extracted from Reddit's API (submissions), preprocess the text and perform several unsupervised exploratory analysis using different libraries (gensim, sklearn...) and approaches (LDA, LSA...)

## Usage

This repository uses **Anaconda** to manage all dependencies:

* If not installed go to and select your version: https://www.anaconda.com/products/distribution
  * Make sure *\<anaconda install path>/* and *\<anaconda install path>/Scripts* are added to your environment variables
  * If there are SSL issues refer to: https://stackoverflow.com/questions/50125472/issues-with-installing-python-libraries-on-windows-condahttperror-http-000-co (top answer) 
* For a simple use guide refer to: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

To create the environment with all the required dependencies execute:

```
conda env create --file environment.yml
```

And then run:

```
conda activate reddit-topic-modelling
```

_Note_: the files to be provided in the 'data' folder are expected to be in JSONL format (one valid JSON per line) and at least containt the fields 'title' and 'selftext'. In addition, the name of the files is expected to contain at least the EXACT name of the subreddit to be used as control (the name of this subreddit is expected to be passed as an argument to the 'main' function)

Example:

```
python main.py ImmigrationCanada
```
