# 数据加载 数据预处理

# 加载句子集
import codecs
import data_utils


def load_sentences(path):
    """
    加载数据集，每一行至少包含一个汉字和一个标记
    句子和句子之间是空格分割
    最后返回距自己和
    :param path:
    :return:
    """
    # 存放数据集
    sentences = []
    # 临时存放每一个句子
    sentence = []
    for line in codecs.open(path, 'r', encoding='utf-8'):
        # 去掉两边空格
        line = line.strip()  # 去除首尾空格
        # 首先判断是否为空  如果是则表示句子和句子之间的分割点
        if not line:  # lin
            if len(sentence) > 0:
                sentences.append(sentence)
                # 清空临时存放的sentence 表示一句话完结
                sentence = []
        # 如果是非空
        else:
            if line[0] == "":
                continue
            else:
                word = line.split()  # 默认是空格
                assert len(word) >= 2
                sentence.append(word)
    # 循环完成，要判断一下，放置最后一个句子没有进入到集合中
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    更新为制定编码 BIO->BIOES
    :param sentences:
    :param tag_scheme:
    :return:
    """
    # 首先是检测编码是否为BIO
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # 首先是检测编码是否为BIO
        if not data_utils.check_bio(tags):
            s_str = "\n".join(" ".join(w) for w in s)  # "\n".join(" ".join(w) for w in s)
            raise Exception("输入的句子应为BIO编码，请检查输入句子%i:\n%s" % (i, s_str))

        if tag_scheme == "BIO":
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag

        if tag_scheme == "BIOES":
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception("非法目标编码！既不是BIO也不是BIOES")


def word_mapping(sentences):
    """
    mapping 映射
    构建字典 word_to_id, id_to_word
    :param sentences:
    :return:
    """
    word_list = [[x[0] for x in s] for s in sentences]
    dico = data_utils.create_dico(word_list)
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = data_utils.create_mapping(dico)
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    构建标签字典
    :param sentences:
    :return:-
    """
    tag_list = [[x[1] for x in s] for s in sentences]
    dico = data_utils.create_dico(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(dico)
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理，返回list其中包含
    - word_list
    - word_id_list
    - word char indexs
    - tag_id_list
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param result:
    :return:
    """
    none_index = tag_to_id['O']
    data = []
    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[x if x in word_to_id else '<UNK>'] for x in word_list]
        segs = data_utils.get_seg_features("".join(word_list))
        if train:
            word_index_list = [tag_to_id[w[-1]] for w in s]
        else:
            word_index_list = [none_index for w in s]
        data.append([word_list, word_id_list, segs, word_index_list])
    return data


# 测试 load_sentences
if __name__ == "__main__":
    path = "data/ner.dev"
    sentences = load_sentences(path)
    update_tag_scheme(sentences, "BIOES")
    _, word_to_id, id_to_word = word_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    dev_data = prepare_dataset(sentences, word_to_id, tag_to_id)
    data_utils.BatchManager(dev_data, 120)
