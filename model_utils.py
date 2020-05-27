from collections import OrderedDict
import os
import json


def config_model(FLAGS, word_to_id, tag_to_id):
    # 有序的字典
    config = OrderedDict()
    config['num_words'] = len(word_to_id)
    config['word_dim'] = FLAGS.word_dim
    config['num_tags'] = len(tag_to_id)
    config['seg_dim'] = FLAGS.seg_dim
    config['lstm_dim'] = FLAGS.lstm_dim
    config['batch_size'] = FLAGS.batch_size
    config['optimizer'] = FLAGS.optimizer
    config['emb_file'] = FLAGS.emb_file

    config['clip'] = FLAGS.clip
    config['dropout_keep'] = 1.0 - FLAGS.dropout
    # 优化器
    config['optimizer'] = FLAGS.optimizer
    # 学习率
    config['lr'] = FLAGS.lr
    config['tag_schema'] = FLAGS.tag_schema
    config['pre_emb'] = FLAGS.pre_emb
    return config


# 1、创建文件夹
def make_path(params):
    a = params.ckpt_path
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir('log'):
        os.makedirs('log')


# 2、保存配置文件
def save_config(config, config_file):
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


# 3、加载配置文件
def load_config(config_file):
    with open(config_file, encoding='utf-8')as f:
       return json.loads(f)