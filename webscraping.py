import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
from googlesearch import search
import csv

#make more sentences for Audrey
#tell visible elements from invisible ones manually
def mask_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(mask_visible, texts)
    '''
    for x, i in enumerate(visible_texts):
        print(i)
        if x>5:
            break
    '''
    return u" ".join(t.strip() for t in visible_texts)

'''setup selenium'''
#print(os.environ['PATH'])
os.environ['PATH'] += r';D:/Selenium_webautomation_drivers'
options = Options()
options.headless = False
driver = webdriver.Chrome(options=options)

#googlesearch query
#breakfast, the first response brings us to a good connection with japan '朝ご飯', https://www.kurashiru.com/lists/d5d8b53c-5cf2-4c4b-b623-9f95ca0666ab
#the problem is that curated information souces, like geeksforgeeks often lack detailed or up to date information.
query = '朝ご飯'
#response = search(query, tld='co.in', num = 10, stop = 10, pause = 2)
response = search(query, tld='co.in', pause = 2)
f = open('storage.csv', 'w', encoding='utf8')
for x,i in enumerate(response):
    print(x, i)
    if True:
        driver.get(i)
        text = text_from_html(driver.page_source).split(' ')
        print('\n gotten text: ', type(text[:500]), '\n')
        #print('\n', 'Source:    ',  driver.page_source[:300],'\n')
        f.writerows(text)
    if x > 3:
        print('finished')
        break
f.close()


driver.quit()
