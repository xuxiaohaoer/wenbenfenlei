from gensim import corpora


def build_dic():

    texts = []
    file =open("./data/wulixuefenci.txt","r",encoding="UTF-8")
    file_data =file.readlines();
    for row in file_data:
        words = row.split(" ")
        words.remove("\n")
        if(len(words)>0 and words[0]!=""):
            it=iter(words)
            for word in words[:]:
                if(word==""):
                    words.remove(word)
                    print(words)
            texts.append(words)
    dic = corpora.Dictionary(texts)
    dic.filter_extremes(keep_n=100000)
    dic.save("test")
    dic.save_as_text("test.txt")
    return dic
dic = build_dic()
print(dic.get(577)+" "+dic.get(578))

