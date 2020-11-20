import os


def get_alltexts(base_dir):
    texts = []
    for filename in os.listdir(base_dir):
        texts += get_texts(base_dir + filename)
    return texts


def get_texts(fullname):
    texts = []
    file = open(fullname, "r", encoding="UTF-8")
    file_data = file.readlines()
    for row in file_data:
        # print(row)
        words = row.split(" ")
        words.remove("\n")
        if (len(words) > 0 and words[0] != ""):
            for word in words[:]:
                if word == "":
                    words.remove(word)
            texts.append(words)
    return texts



if __name__ == '__main__':
    text = get_texts("./data/zhongguogudaishifenci.txt")
    print(len(text))
