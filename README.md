# reddit-topic-modelling

## Characteristics

Given JSON files extracted from Reddit's API (submissions), preprocess the text and perform several unsupervised exploratory analysis using different libraries (gensim, sklearn...) and approaches (LDA, LSA...)

## Usage

To create the environment with all the required dependencies execute:

```
conda env create --file environment.yml
```

And then run:

```
conda activate reddit-topic-modelling
```

_Note_: the files to be provided in the 'data' folder are expected to be in JSONL format (one valid JSON per line) and at least containt the fields 'title' and 'selftext'. In addition, the name of the files is expected to contain at least the EXACT name of the subreddit to be used as control (the name of this subreddit is expected to be passed as an argument to the 'function')

Example:

```
python main.py ImmigrationCanada
```