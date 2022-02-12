from googletrans import Translator
import string

translator = Translator(service_urls=['translate.googleapis.com'])
translator = Translator()

transSentence = input("Enter Sentence To Be Translated: ")

language = input("Enter language to be translated to: ")

vocabList = transSentence.split(" ")

for item in vocabList:
    for i in item:
        if i in string.punctuation:
            item1 = item.replace(i,"")
            vocabList.remove(item)
            vocabList.append(item1)

definition = ""
lang1 = translator.detect(transSentence)

print("{:<10} {:<10}".format(lang1.lang,language))

for item in vocabList:
    definition = translator.translate(item, dest=language)
    print("{:<10} {:<10}".format(item,definition.text))
