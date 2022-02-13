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
def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)
    print(result)
    print(u"Text: {}".format(result["input"]))
    print(u"Translation: {}".format(result["translatedText"]))
    print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))

def translateEnglish(sentence):

    #translator = Translator(service_urls=['translate.googleapis.com'])
    translator = Translator()
    print('heeeeeee',translator.detect(sentence))
    pronunciation = translator.translate(sentence, dest = translator.detect(sentence).lang)
    english = translator.translate(sentence, src = 'ja', dest = 'en')
    print(english)
    return english.text, pronunciation.pronunciation
