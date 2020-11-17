import random

from getext import get_texts

catagorys = ["hulianwang", "jinrong", "junshi", "makesi", "qichegongye", "tiyu", "wulixue", "xinlixue", "yuanyi",
             "zhongguogudaishi"]
base_dir = "./data/"
words = []
random.seed(1)
train_size = 50000  # 每个类别中训练集大小


def loda_file(base_dir):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for catagory in catagorys:
        print("读取类别" + catagory + "中")
        texts = get_texts(base_dir + catagory + "fenci.txt")
        random.shuffle(texts)
        x_train += texts[:train_size]
        y_train += [catagorys.index(catagory)] * train_size
        # 将类别的种类转化为数字索引
        x_test += texts[train_size:]
        y_test += [catagorys.index(catagory)] * (len(texts) - train_size)
    print("读取完成")
    dataset = x_train + x_test
    # 完整的数据集
    test = dataset[0:100]
    # 测试功能用的数据集？
    return x_train, y_train, x_test, y_test, dataset, test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, dataset, x_test = loda_file(base_dir)
    print(len(x_train))
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))
    print(len(dataset))
