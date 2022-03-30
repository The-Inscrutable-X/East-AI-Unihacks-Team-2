import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webscraping import parse_another_site, search
from simple_translator import translateEnglish
from sentence_segmentation import display_separated
from understandability_algorithm import Understandability

"""setup selenium"""
print('setting up selenium')
os.environ['PATH'] += r';C:\Users\chenz\Documents\GitHub\East-AI-Unihacks-Team-2\chromewebdriver'
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)
print('finished setting up selenium')
#import PySimpleGUI as pg

#"""setup gui"""

#window = sg.Window(title="Hello World", layout=[[]], margins=(100, 50)).read()
"""
examples, googlesearch querys:
breakfast, the first response brings us to a good connection with japan '朝ご飯', https://www.kurashiru.com/lists/d5d8b53c-5cf2-4c4b-b623-9f95ca0666ab
the problem is that curated information souces, like geeksforgeeks often lack detailed or up to date information.
english test query = lightning,
Alt: 朝ごはん,
Result: query: 朝ご飯, sentence: 1000人が絶賛の朝ご飯レシピ, trans: Breakfast recipe acclaimed by 1000 people
We may be able to reroute searches to japanese reddit

Interesting Results Archive:
行き先: 【合唱曲】行き先 / 歌詞付き: [Chorus] Destination / with lyrics: Youtube Video
めいわくでんわ: 迷惑電話ストップサービス: Prank call stop service: Japan has a scam service stop service? America does not have this.
"""
def weblang(query_origin, language = 'de', target_sentences = 10):

    input('buffer')
    language = 'de'
    # query_origin = "Mann kommt"
    query = '"'+query_origin+'"'
    # target_sentences = 5
    target_understandability = 1.75
    #query = query_origin

    print('ai training start')
    #response = search(query, tld='co.in', num = 10, stop = 10, pause = 2)
    understandability_algorithm = Understandability('data_to_train.csv', debug = False)
    understandability_algorithm.train()
    print('ai training_done')
    #print('Class testing:', understandability_algorithm.predict("vocabs are ontime and dazzling and fantastic."))
    #quit()
    response = search(query, pause = 2, num = 30, stop = 30, lang = language)
    #response = search(query, tld='co.in', pause = 2)
    with open('storage.csv', 'w', encoding='utf8') as f:
        good_sentences = 0
        output_sentences = []
        data_sentences = []
        for x in range(30):
            if good_sentences >= target_sentences:
                break
            print(x)
            sentences, url = parse_another_site(response, driver, f, query_origin)
            try:
                parse_limit = int(input('how many sentences to review: '))
                mode = input('improvement mode, y/n: ')
            except:
                print('skipping this website')
                parse_limit = 0
            for x, sentence in enumerate(sentences):
                if x >= parse_limit:
                    break

                if mode == 'y':

                    print('human guidance mode')
                    print('|uncovered sentence:', sentence, '\n|url', url )
                    understandability = float(input('understandability of this sentence: '))
                    data_sentences.append((sentence, understandability))
                    if abs(understandability-target_understandability) < .5:
                        output_sentences.append((sentence, url))
                        good_sentences += 1
                else:
                    print('automatic mode')
                    output_sentences.append((sentence, url))
                    good_sentences += 1

                api_broken = True
                if api_broken == False:
                    converted_sentence, converted_sentence_pronounciation = translateEnglish(sentence)
                    """with open('storage.txt', 'a', encoding='utf8') as g:
                        g.writelines('|original', sentence, '\n|translated', converted_sentence, '\n|pronounciation', converted_sentence_pronounciation, '\n')
                        pass"""
                    if converted_sentence_pronounciation == None:
                        score = understandability_algorithm.predict(converted_sentence)
                    elif converted_sentence_pronounciation != None:
                        score = understandability_algorithm.predict(converted_sentence_pronounciation)

                    if score == 1:
                        output_sentences.append([sentence, converted_sentence, converted_sentence_pronounciation, url, score])
                        good_sentences += 1
                else:
                    score = understandability_algorithm.predict(sentence)
                    print(score)
                    if score == 1:
                        output_sentences.append([sentence, converted_sentence, converted_sentence_pronounciation, url, score])
                        good_sentences += 1



        print('finished')
        #converted_sentence, converted_sentence_pronounciation = translate_text('en', sentence)
        try:
            with open('output.txt', 'a+', encoding='utf8') as f:
                f.write('\n\n')
                f.write('\n'.join([str(i) for i in output_sentences]))
            checkout = output_sentences[0][1]
            checkout_sentence = output_sentences[0][0]
            driver.get(checkout)
            import pyperclip
            pyperclip.copy(checkout_sentence)
            spam = pyperclip.paste()
            #print(spam)
            if api_broken == False:
                converted_sentence, converted_sentence_pronounciation = translateEnglish(sentence)
                print('\n|original', sentence, '\n|translated', converted_sentence, '\n|pronounciation', converted_sentence_pronounciation, '\n|url', url, '\n|comprehension level', score)
                display_separated(converted_sentence_pronounciation, 'en')
        except IndexError:
            print('query busted, do not include underlines or special formats, query must exist in website text exactly')
        input('close webdriver? ')
        driver.quit()
        if input('update ml algorithm with new information from this study session? y/n: ') == 'n':
            print('Exiting')
        else:
            understandability_algorithm.update(data_sentences)






def simple_weblang(query_origin = "Mann kommt", language = 'de', target_sentences = 3):

    query = '"'+query_origin+'"'
    parse_limit = 5 #limit of sentences to source from one website
    #target_understandability = 1.75

    print('ai training start, new session started')
    #response = search(query, tld='co.in', num = 10, stop = 10, pause = 2)
    understandability_algorithm = Understandability('data_to_train.csv', debug = False)
    understandability_algorithm.train()
    print('ai training_done')
    #print('Class testing:', understandability_algorithm.predict("vocabs are ontime and dazzling and fantastic."))
    #quit()
    response = search(query, pause = 2, num = target_sentences, stop = target_sentences, lang = language)
    #response = search(query, tld='co.in', pause = 2)
    with open('storage.csv', 'w', encoding='utf8') as f:
        good_sentences = 0
        output_sentences = []
        data_sentences = []
        for x in range(30):
            if good_sentences >= target_sentences:
                break
            print('\nwebsite number', x)
            sentences, url = parse_another_site(response, driver, f, query_origin)
            if parse_limit > target_sentences:
                parse_limit = target_sentences

            for x, sentence in enumerate(sentences):
                if x >= parse_limit:
                    print('breaking and current x', x, 'limit', parse_limit)
                    break

                api_broken = True
                if api_broken == False:
                    converted_sentence, converted_sentence_pronounciation = translateEnglish(sentence)
                    """with open('storage.txt', 'a', encoding='utf8') as g:
                        g.writelines('|original', sentence, '\n|translated', converted_sentence, '\n|pronounciation', converted_sentence_pronounciation, '\n')
                        pass"""
                    if converted_sentence_pronounciation == None:
                        score = understandability_algorithm.predict(converted_sentence)
                    elif converted_sentence_pronounciation != None:
                        score = understandability_algorithm.predict(converted_sentence_pronounciation)

                    if score == 1:
                        output_sentences.append([sentence, converted_sentence, converted_sentence_pronounciation, url, score])
                        good_sentences += 1
                else:
                    score = understandability_algorithm.predict(sentence)
                    print(score)
                    if score > 0:
                        output_sentences.append([sentence, url, score])
                        good_sentences += 1

        print('finished')
        #converted_sentence, converted_sentence_pronounciation = translate_text('en', sentence)
        try:
            with open('output.txt', 'w', encoding='utf8') as f:
                f.write('\n\n')
                f.write('\n'.join([str(i) for i in output_sentences]))
            checkout = output_sentences[0][1]
            checkout_sentence = output_sentences[0][0]
            driver.get(checkout)
            import pyperclip
            pyperclip.copy(checkout_sentence)
            spam = pyperclip.paste()
            #print(spam)
            if api_broken == False:
                converted_sentence, converted_sentence_pronounciation = translateEnglish(sentence)
                print('\n|original', sentence, '\n|translated', converted_sentence, '\n|pronounciation', converted_sentence_pronounciation, '\n|url', url, '\n|comprehension level', score)
                display_separated(converted_sentence_pronounciation, 'en')
        except IndexError:
            print('query busted, do not include underlines or special formats, query must exist in website text exactly')
        driver.quit()
        understandability_algorithm.update(data_sentences)
    print('returning output')
    return output_sentences

#weblang('行き先')
print('\n\n\n',"\n".join([str(i) for i in simple_weblang(query_origin = 'Mann kommt')]))
