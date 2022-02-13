from webscraping import *
from simple_translator import *
from api2 import *
from understandability_algorithm import Understandability
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
english query = lightning
"""
query_origin = '朝ご飯'
query = '"'+query_origin+'"'
#query = query_origin

#response = search(query, tld='co.in', num = 10, stop = 10, pause = 2)
understandability_algorithm = Understandability('data_to_train.csv', debug = True)
understandability_algorithm.train()
print('Class testing:', understandability_algorithm.predict("hi how are you"))
quit()

response = search(query, tld='co.in', pause = 2)
with open('storage.csv', 'w', encoding='utf8') as f:
    good_sentences = 0
    output_sentences = {}
    for x in range(1):
        print(x)
        sentences, url = parse_another_site(response, driver, f, query_origin)
        #for x, sentence in enumerate(sentences):
            #   if good_sentences > 3:
            #       break
            #converted_sentence, converted_sentence_pronounciation = translateEnglish(sentence)
            #
            #print(sentence, converted_sentence, converted_sentence_pronounciation)
            #score = understandability_algorithm(converted_sentence)
            #if score>.7:
            #   output_sentences[sentence] = [converted_sentence, url, ]
            #   good_sentences += 1

            #if x>0:
            #    break

    else:
        print('finished')
        sentence = sentences[0]
        #converted_sentence, converted_sentence_pronounciation = translate_text('en', sentence)
        converted_sentence, converted_sentence_pronounciation = translateEnglish(sentence)
        print('\n|', sentence, '\n|', converted_sentence, '\n|', converted_sentence_pronounciation)
        display_separated(converted_sentence_pronounciation, 'en')


#driver.quit()
