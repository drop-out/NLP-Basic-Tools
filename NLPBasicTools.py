
#coding=utf-8

# seperate a word into a list of ngrams
def ngram_generator(word,gram_length=2,seperator=''):
    if seperator is None:
        seperator = ''
    if word is None:
        return []
    if seperator!='':
        gram_list = word.split(seperator)
    else:
        gram_list = list(word)
        
    if len(gram_list)<gram_length:
        return [seperator.join(gram_list)]
    
    result = []
    for i in range(0,len(gram_list)-gram_length+1):
        result.append(seperator.join(gram_list[i:i+gram_length]))
    return result

# calculate elements in the intersection of two ngram list
def ngram_intersection_unique(ngram1,ngram2):
    intersect_grams = set(ngram1)&set(ngram2)
    return len(intersect_grams)

# calculate elements in union of two ngram list
def ngram_union_unique(ngram1,ngram2):
    union_grams = set(ngram1)|set(ngram2)
    return len(union_grams)
