from googletrans import Translator
import string
import pandas as pd

translator = Translator(service_urls=['translate.googleapis.com'])
translator = Translator()

transSentence = input("Enter Sentence To Be Translated: ")

language = input("Enter language to be translated to: ")

#japanese word segmentation
#https://japanese.stackexchange.com/questions/11687/how-to-separate-words-in-a-japanese-sentence#:~:text=Separating%20words%20in%20a%20sentence%2C%20at%20least%20when,dictionary%20to%20replace%20the%20kana%20with%20its%20kanji.
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

#divider1 = lang1.lang.length()
#divider2 = language.length()



#print("{:<10} {:<10}".format())

definitions = []

for item in vocabList:
    definition = translator.translate(item, dest=language)
    definitions.append(definition)
    print("{:<10} {:<10}".format(item,definition.text))

#df = pd.DataFrame(d, columns)
