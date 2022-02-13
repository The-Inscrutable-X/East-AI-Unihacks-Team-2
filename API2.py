from googletrans import Translator
import string

def display_separated(transSentence, language):
    translator = Translator(service_urls=['translate.googleapis.com'])
    translator = Translator()

    languages = ['Afrikaans (af)', 'Albanian (sq)', 'Amharic (am)', 'Arabic (ar)', 'Armenian (hy)', 'Azerbaijani (az)', 'Basque (eu)', 'Belarusian (be)', 'Bengali (bn)', 'Bosnian (bs)', 'Bulgarian (bg)', 'Catalan (ca)', 'Cebuano (ceb)', 'Chichewa (ny)', 'Chinese (Simplified) (zh-CN)', 'Chinese (Traditional) (zh-TW)', 'Corsican (co)', 'Croatian (hr)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Esperanto (eo)', 'Estonian (et)', 'Filipino (tl)', 'Finnish (fi)', 'French (fr)', 'Frisian (fy)', 'Galician (gl)', 'Georgian (ka)', 'German (de)', 'Greek (el)', 'Gujarati (gu)', 'Haitian Creole (ht)', 'Hausa (ha)', 'Hawaiian (haw)', 'Hebrew (iw)', 'Hindi (hi)', 'Hmong (hmn)', 'Hungarian (hu)', 'Icelandic (is)', 'Igbo (ig)', 'Indonesian (id)', 'Irish (ga)', 'Italian (it)', 'Japanese (ja)', 'Javanese (jw)', 'Kannada (kn)', 'Kazakh (kk)', 'Khmer (km)', 'Kinyarwanda (rw)', 'Korean (ko)', 'Kurdish (Kurmanji) (ku)', 'Kyrgyz (ky)', 'Lao (lo)', 'Latin (la)', 'Latvian (lv)', 'Lithuanian (lt)', 'Luxembourgish (lb)', 'Macedonian (mk)', 'Malagasy (mg)', 'Malay (ms)', 'Malayalam (ml)', 'Maltese (mt)', 'Maori (mi)', 'Marathi (mr)', 'Mongolian (mn)', 'Myanmar (Burmese) (my)', 'Nepali (ne)', 'Norwegian (no)', 'Odia (Oriya) (or)', 'Pashto (ps)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Punjabi (pa)', 'Romanian (ro)', 'Russian (ru)', 'Samoan (sm)', 'Scots Gaelic (gd)', 'Serbian (sr)', 'Sesotho (st)', 'Shona (sn)', 'Sindhi (sd)', 'Sinhala (si)', 'Slovak (sk)', 'Slovenian (sl)', 'Somali (so)', 'Spanish (es)', 'Sundanese (su)', 'Swahili (sw)', 'Swedish (sv)', 'Tajik (tg)', 'Tamil (ta)', 'Tatar (tt)', 'Telugu (te)', 'Thai (th)', 'Turkish (tr)', 'Turkmen (tk)', 'Ukrainian (uk)', 'Urdu (ur)', 'Uyghur (ug)', 'Uzbek (uz)', 'Vietnamese (vi)', 'Welsh (cy)', 'Xhosa (xh)', 'Yiddish (yi)', 'Yoruba (yo)', 'Zulu (zu)', 'Hebrew (he)', 'Chinese (Simplified) (zh)']
    '''
    question = input("Would you like to see a table of possible languages? (y/n): ")
    if question == "y":
        for item in languages:
            print(item)

    transSentence = input("Enter Sentence To Be Translated: ")

    language = input("Enter language to be translated to: ")
    '''
    vocabList = transSentence.split(" ")

    for item in vocabList:
        for i in item:
            if i in string.punctuation:
                item1 = item.replace(i,"")
                print('punctuation: ',item1,item)
                vocabList.remove(item)
                vocabList.append(item1)

    definition = ""

    definitions = []

    lang1 = translator.detect(transSentence)

    languageName = ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Bulgarian', 'Catalan', 'Cebuano', 'Chichewa', 'Chinese', 'Chinese', 'Corsican', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Esperanto', 'Estonian', 'Filipino', 'Finnish', 'French', 'Frisian', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Korean', 'Kurdish', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Myanmar', 'Nepali', 'Norwegian', 'Odia', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Samoan', 'Scots Gaelic', 'Serbian', 'Sesotho', 'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu', 'Hebrew', 'Chinese']

    languageSymbol = ['af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'ny','zh-CN','zh-TW', 'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'tl', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 'km', 'rw', 'ko','ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn','my', 'ne', 'no','or', 'ps', 'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'tt', 'te', 'th', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu', 'he','zh']

    for item in languageSymbol:
        if lang1.lang in languageSymbol:
            languageReturn = languageName[languageSymbol.index(lang1.lang)]

    if len(language) <= 3:
        if language in languageSymbol:
            languageReturn2 = languageName[languageSymbol.index(language)]

    print("\nVocab Table: \n")

    print("{:<10} {} {:<10}".format(languageReturn,"|",languageReturn2))
    print("{:<10} {} {:<10}".format("----------","|","----------"))

    for item in vocabList:
        definition = translator.translate(item, dest=language)
        definitions.append(definition.text)
        print("{:<10} {} {:<10}".format(item,"|",definition.text))

    #print(vocabList, definitions)

    #Would you like to see a table of possible languages? (y/n): n
    #Enter Sentence To Be Translated: hola mi nombre es genial
    #Enter language to be translated to: af
    #Spanish    | Afrikaans
    #---------- | ----------
    #hola       | hallo
    #mi         | ek
    #nombre     | Naam
    #es         | is
    #genial     | koel
