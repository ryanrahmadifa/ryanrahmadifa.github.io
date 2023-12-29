---
layout: page
title: market research
description: Selenium, BeautifulSoup, deep-translator, HuggingFace, GPT-3.5
img: assets/img/nlp-market-research-thumbnail.png
importance: 3
category: natural language processing
---

This project proposes a market research system using state-of-the-art libraries for NLP, with Python as a base for automation of every process in the pipeline, demonstrating its potential for real-world implementations for businesses.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/nlp-market-research-thumbnail.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Flowchart of the process of the project.
</div>

NLP is largely implemented by using the Python programming language and various open-source libraries that are available for Python. 

The implementations of NLP in this essay are in the forms of:
1. machine translation with deep-translator for translating data based on Bahasa Indonesia to English, continued with 
2. topic modelling with BERTopic towards Google News and Google Play Store review data to extract the main ideas of both data
3. sentiment analysis with BERT for gauging receptiveness from the government and the large public towards EVs, and lastly 
4. idea generation through GPT-3.5 subject to all the previous data for contextual alternatives on business strategies for MaaS. 

Web scraping (Mahto & Singh, 2016) is the chosen method for primary data collection in this project, based on the goal of achieving large amounts of data collection effectively and efficiently, this project also uses Selenium (Muthukadan, 2023) for developing browser automation and BeautifulSoup (Richardson, 2007) to scrape Google News based on a predetermined query through HTML parsing.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/nlp-market-research-results.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results at the end of the pipeline (idea generation via GPT-3.5)
</div>