# Drivers-Article-Matching
Drivers to Article and Article to Driver Matching for the SPA project

## Link to Model Weights
https://rpi.box.com/s/ym36qaofw2rk93h3t7zbpnphu25p0q7z

## Usage
Example use is shown in `match_fn.py`
The input to the `Ranker` object are the directories containing the weights and the vocab file.
For query->article, use query2articles function.
For article->query, use article2queries function.
The query and article must be lowercased and tokenized and fed in as a list of tokens.
