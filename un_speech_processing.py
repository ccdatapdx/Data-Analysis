import os
import re
import spacy
from spacy.tokens import Doc
import pandas as pd


file_names = []
speeches = []

for root, dirs, files in os.walk('/home/ccpdx/projects/un_speeches_nlp_explorer/TXT'):
    for name in files:
        if name.endswith('.txt') and bool(re.search('^\w',name)):
            file_names.append({'speech_id':name})
            with open(os.path.join(root,name),encoding='utf-8') as f:
                txt_file = f.read()
                txt_file = ''.join(txt_file)
                speeches.append(txt_file)

speeches = list(zip(speeches,file_names))



if not Doc.has_extension("speech_id"):
    Doc.set_extension("speech_id", default=None)

entities_text = []
entities_label = []
context_text = []

nlp = spacy.load("en_core_web_lg")

for doc,context in nlp.pipe(speeches, as_tuples=True,
                            disable=["tok2vec", "tagger", "parser",
                                    "attribute_ruler", "lemmatizer"], 
                                     n_process=-1,batch_size=20):
    doc._.speech_id = context['speech_id']
    for test in doc.ents:
        entities_text.append(test.text)
        entities_label.append(test.label_)
        context_text.append(doc._.speech_id)

data = {'entities':entities_text,'label':entities_label,'context':context_text}

data = pd.DataFrame(data)

final_data = data.to_parquet('processed_un_speeches.parquet')

final_data

