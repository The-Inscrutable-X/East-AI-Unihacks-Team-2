import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
from googlesearch import search
import csv
import re
from time import sleep


#make more sentences for Audrey
#tell visible elements from invisible ones manually
def mask_visible(element):
    #element.is_displayed()
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(html, query):
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    output = ''
    '''blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        # there may be more elements you don't want, such as "style", etc.
    ]'''
    for t in texts:
        if mask_visible(t):
            #add a newline if previous or next is hyperlink
            '''if t.parent.name == 'a':
                output += str(t)
            elif last_t_type == 'a':
                for s in re.split('\.', t):
                    output += str(t)+'|'
            elif t.parent.name != 'a':
                for s in re.split('\.', t):
                    if len(s)>20:
                        output += str(t)+'|'
            last_t_type = t.parent.name'''
            #for s in re.split('\.', t):
            #    output += str(t)+'|'
            output += '\n{}'.format(t)

    '''visible_texts = (i.get_text() for i in texts)
    for x,i in enumerate(visible_texts):
        print(i)
        if x>5:
            break
    print('type', type(texts))
    def generator(texts):
        for t in texts:
            pass'''
    print('html text areas found:', len(texts))
    #((s+'.').strip() for t in visible_texts if (t != '\n') for s in re.split('\.', t) if (len(s)>20 and t.parent.name != 'a'))
    #return [i.strip() for i in re.split('\.|\n|。', output) if len(i.strip())>20]
    return [i.strip() for i in re.split('\.|\n|。', output) if len(i.strip())>len(query)]

def parse_another_site(response_object, driver, f, query):
    url = next(response_object)
    driver.get(url)
    sleep(3)
    text = text_from_html(driver.page_source, query)

    print('\n gotten text: ', type(text), 'Sentence count: ', len(text), url, '\n')
    for i in text:
        f.writelines('"'+i+'",\n')

    output = [i for i in text if (query in i)]
    print('how many acceptable sentences were found: ', len(output))
    return output, url
