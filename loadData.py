import tarfile
import re


def load_imdb(is_training):
    text_set = []
    label_set = []

    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的rarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里

    for label in ["pos", "neg"]:
        with tarfile.open("./aclImdb_v1.tar.gz") as tarf:  # 打开压缩包
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)  # 生成正则对象
            tf = tarf.next()  # 显示下一个文件信息
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    # print('tf.name:',tf.name)
                    sentence = tarf.extractfile(tf).read().decode()  # 从tf文件中提取文件，读取内容并解码
                    sentence_label = 0 if label == 'neg' else 1
                    text_set.append(sentence)
                    label_set.append(sentence_label)
                tf = tarf.next()

    return text_set, label_set


train_text, train_label = load_imdb(True)
test_text, test_label = load_imdb(False)
