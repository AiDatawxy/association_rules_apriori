# Association Rules - Apriori Algorithme

* this is a simple implementation of apriori algorithm to mine association rules in DataFrame, using python generation to handle large data set

* apriori algorithm in python package require unacceptable memory for large data set. Here refering [the article](https://www.datatheque.com/posts/association-analysis/), I implement the alogrithm taking advantage of generation and iterators in python, which save a lot of memory; the source aritcle implement the algorithm for frequency items with as many as 2 goods, here i extended it to frequency items with more than 2 goods

* besides, I use this method to analyze DataFrame data, first discretize numerical varialbes and second transform DataFrame data to transaction data set; then applying the apriori to mine rules

* [data_feature.py](https://github.com/AiDatawxy/association_rules_apriori/blob/master/data_feature.py) contains functions to pre-handle data set, and [apriori.py](https://github.com/AiDatawxy/association_rules_apriori/blob/master/apriori.py) contains the main algorithm
