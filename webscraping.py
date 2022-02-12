import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
from googlesearch import search
import csv
import re

#make more sentences for Audrey
#tell visible elements from invisible ones manually
def mask_visible(element):
    #element.is_displayed()
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(mask_visible, texts)
    print('type', type(texts))
    def generator(texts):
        for t in texts:

        pass
    return ((s+'.').strip() for t in visible_texts if (t != '\n') for s in re.split('\.', t) if len(s)>20)

def parse_another_site(response_object, driver, f):
    url = next(response_object)
    driver.get(url)
    text = text_from_html(driver.page_source)

    #print('\n gotten text: ', url, type(text), 'Sentence count: ',len(text), '\n')
    for i in text:
        f.writelines('"'+i+'",\n')
    pass

"""
setup selenium
"""
#print(os.environ['PATH'])
os.environ['PATH'] += r';D:/Selenium_webautomation_drivers'
options = Options()
options.headless = False
driver = webdriver.Chrome(options=options)

"""
googlesearch query
breakfast, the first response brings us to a good connection with japan '朝ご飯', https://www.kurashiru.com/lists/d5d8b53c-5cf2-4c4b-b623-9f95ca0666ab
the problem is that curated information souces, like geeksforgeeks often lack detailed or up to date information.
"""

query = 'lightning'
#response = search(query, tld='co.in', num = 10, stop = 10, pause = 2)
response = search(query, tld='co.in', pause = 2)
with open('storage.csv', 'w', encoding='utf8') as f:
    #csv_writer = csv.writer(f)
    for x in range(3):
        print(x)
        parse_another_site(response, driver, f)
    else:
        print('finished')

driver.quit()
