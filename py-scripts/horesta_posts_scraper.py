#!/usr/bin/env python
# coding: utf-8

# NOTE: This script uses a chromedriver (with its head) to scrape links to articles. 
# The script is therefore very OS dependent and may not work in other environments in its current state.

import requests

import os
import sys

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, WebDriverException, ElementNotInteractableException
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup as bs

import random
import re
import time
from datetime import datetime as dt
import json


# Path for storing data and urls
data_path = os.path.join('..', 'data')


# Loading and starting chrome webdriver
driver = webdriver.Chrome(executable_path = 'C:\\chromedriver\\chromedriver.exe')

driver.get('https://www.horesta.dk/nyheder/')
time.sleep(3)


# Coords for reaching p.btn-load-more button
x_coord = 0
y_coord = 2400


# Scrolls through posts until p.btn-load-more button no longer shows up (no more posts)
while True:
    coordinates = {'x': x_coord, 'y': y_coord}
    driver.execute_script('window.scrollTo({}, {});'.format(coordinates['x'], coordinates['y']))
    try:
        driver.find_element_by_css_selector("p.btn-load-more").click()
        y_coord = y_coord + 2400
        time.sleep(1)
    except NoSuchElementException:
        break

        
# Store the entire loaded page source as one html document        
pageSource = driver.page_source


# Quit the chome driver
driver.quit()


# Convert pagesource to soup object
pagesoup = bs(pageSource)


# Extract links to posts
links = [s.find('a')['href'] for s in pagesoup.find_all('div', class_ = 'post-item')]


# Store links as txt
with open(os.path.join(data_path,'horesta_urls.txt'), 'w', encoding = 'utf-8') as f:
    for link in links:
        f.write(link + "\n")

        
# Defining functions
def get_article_links(soup):
    """
    Retrieves all href attributes from a tags in a soup object.
    """
    
    links = []
    
    for s in soup.find_all('a'):
        try:
            links.append(s['href'])
        except:
            continue
    
    return(links)

def get_article_tags(soup):
    """
    Retrieves the post tags as set by Horesta
    """
    try:
        article_tags = soup.find('ul', class_='tag-list').get_text().strip().split('\n')
    except:
        article_tags = []
        
    return(article_tags)

def get_article_info(url):
    """
    Retrieves article info: url, title, tags, links in article, publish date, article text and source code.
    Also stores dummy variable 'accessed' for whether or not article was accessed as well as date of access.
    Returns as a dictionary with keys: url, accessed, title, tags, links, publish_date, access_date, text, html
    """
    article_dict = {}
    
    response = requests.get(url)
    if response.status_code != 200:
        article_dict['url'] = url
        article_dict['accessed'] = 0
        article_dict['title'] = ""
        article_dict['tags'] = ""
        article_dict['links'] = ""
        article_dict['publish_date'] = ""
        article_dict['access_date'] = str(dt.now().date())
        article_dict['text'] = ""
        article_dict['html'] = ""
    else:
        soup = bs(response.content)
        article_soup = soup.find('article', class_='article-main')
        
        article_dict['url'] = url
        article_dict['accessed'] = 1
        article_dict['title'] = soup.find('h1', class_='margin-top-bottom-0').get_text(strip = True)
        article_dict['tags'] = get_article_tags(soup)
        article_dict['links'] = get_article_links(article_soup)
        article_dict['publish_date'] = soup.find('div', id = 'divDate').find('span').next_sibling.get_text()
        article_dict['access_date'] = str(dt.now().date())
        article_dict['text'] = article_soup.get_text()
        article_dict['html'] = str(soup)
        
    return(article_dict)


# Retrieving articles based on list of links
articles = list()

for c, link in enumerate(links, start = 1):
    link = "https://horesta.dk" + link
    art_dict = get_article_info(link)
    articles.append(art_dict)
      
    print("{:.2f}% of articles downloaded".format(100.0 * c/len(links)), end = '\r')
    
    sleep_time = random.uniform(0.3, 0.9)
    time.sleep(sleep_time)


# Saving articles as json list
with open( os.path.join(data_path,'horesta_posts.json'), 'w', encoding = 'utf-8') as f:
    json.dump(articles, f)

