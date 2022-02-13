from googletrans import Translator
#from langdetect import detect

'''def translate_to_english(sentence):

    translator = Translator(service_urls=['translate.googleapis.com'])

    lang1 = str(detect(sentence)),

    stringer = ''
    for i in lang1:
        stringer+=i

    print('\n\n\n', sentence, stringer, '\n\n\n')
    english = translator.translate(sentence, src=stringer,  dest = 'en')

    return english
'''
def translateEnglish(sentence):

    translator = Translator(service_urls=['translate.googleapis.com'])
    translator = Translator()

    english = translator.translate(sentence, dest = 'en')

    return english.text
