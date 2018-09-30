
'''
diction = corpora.Dictionary([doc.split(' ') for doc in doc_set])
dct.add_documents([["cat", "say", "meow"], ["dog"]])
corpus = [diction.doc2bow(doc.split()) for doc in doc_clean_set]
tfidf = models.TfidfModel(corpus)
'''