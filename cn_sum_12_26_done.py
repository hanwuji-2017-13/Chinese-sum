# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
import jieba
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from gensim.models import Word2Vec
import pprint, pickle
print('TensorFlow Version: {}'.format(tf.__version__))


# 单个文本的 clean 函数
def clean_text(text, remove_stopwords = True):

    
    contractions = {
 	u'吻腚':u'稳定',
 	u'弓虽':u'强',
 	u'女干':u'奸',
 	u'示土':u'社',
 	u'禾口':u'和',
 	u'言皆':u'谐',
 	u'释永性':u'释永信',
 	u'大菊观':u'大局观',
 	u'yl':u'一楼',
 	u'cnm':u'草泥马',
 	u'CCTV':u'中央电视台',
 	u'CCAV':u'中央电视台',
 	u'ccav':u'中央电视台',
 	u'cctv':u'中央电视台',
 	u'qq':u'腾讯聊天账号',
 	u'QQ':u'腾讯聊天账号',
 	u'cctv':u'中央电视台',
 	u'CEO':u'首席执行官',
 	u'克宫':u'克里姆林宫',
 	u'PM2.5':u'细颗粒物',
 	u'pm2.5':u'细颗粒物',
 	u'SDR':u'特别提款权',
 	u'装13':u'装逼',
 	u'213':u'二逼',
 	u'13亿':u'十三亿',
 	u'巭':u'功夫',
 	u'孬':u'不好',
 	u'嫑':u'不要',
 	u'夯':u'大力',
 	u'芘':u'操逼',
 	u'烎':u'开火',
 	u'菌堆':u'军队',
 	u'sb':u'傻逼',
 	u'SB':u'傻逼',
 	u'Sb':u'傻逼',
 	u'sB':u'傻逼',
 	u'is':u'伊斯兰国',
 	u'isis':u'伊斯兰国',
 	u'ISIS':u'伊斯兰国',
 	u'ko':u'打晕',
 	u'你M':u'你妹',
 	u'你m':u'你妹',
 	u'震精':u'震惊',
 	u'返工分子':u'反共',
 	u'黄皮鹅狗':u'黄皮肤俄罗斯狗腿',
 	u'苏祸姨':u'苏霍伊',
 	u'混球屎报':u'环球时报',
 	u'屎报':u'时报',
 	u'jb':u'鸡巴',
 	u'j巴':u'鸡巴',
 	u'j8':u'鸡巴',
 	u'J8':u'鸡巴',
 	u'JB':u'鸡巴',
 	u'瞎BB':u'瞎说',
 	u'nb':u'牛逼',
 	u'牛b':u'牛逼',
 	u'牛B':u'牛逼',
 	u'牛bi':u'牛逼',
 	u'牛掰':u'牛逼',
 	u'苏24':u'苏两四',
 	u'苏27':u'苏两七',
 	u'痰腐集团':u'贪腐集团',
 	u'痰腐':u'贪腐',
 	u'反hua':u'反华',
 	u'<br>':u' ',
 	u'屋猫':u'五毛',
 	u'5毛':u'五毛',
 	u'傻大姆':u'萨达姆',
 	u'霉狗':u'美狗',
 	u'TMD':u'他妈的',
 	u'tmd':u'他妈的',
 	u'japan':u'日本',
 	u'P民':u'屁民',
 	u'八离开烩':u'巴黎开会',
 	u'傻比':u'傻逼',
 	u'潶鬼':u'黑鬼',
 	u'cao':u'操',
 	u'爱龟':u'爱国',
 	u'天草':u'天朝',
 	u'灰机':u'飞机',
 	u'张将军':u'张召忠',
 	u'大裤衩':u'中央电视台总部大楼',
 	u'枪毕':u'枪毙',
 	u'环球屎报':u'环球时报',
 	u'环球屎包':u'环球时报',
 	u'混球报':u'环球时报',
 	u'还球时报':u'环球时报',
 	u'人X日报':u'人民日报',
 	u'人x日报':u'人民日报',
 	u'清只县':u'清知县',
 	u'PM值':u'颗粒物值',
 	u'TM':u'他妈',
 	u'首毒':u'首都',
 	u'gdp':u'国内生产总值',
 	u'GDP':u'国内生产总值',
 	u'鸡的屁':u'国内生产总值',
 	u'999':u'红十字会',
 	u'霉里贱':u'美利坚',
 	u'毛子':u'俄罗斯人',
 	u'ZF':u'政府',
 	u'zf':u'政府',
 	u'蒸腐':u'政府',
 	u'霉国':u'美国',
 	u'狗熊':u'俄罗斯',
 	u'恶罗斯':u'俄罗斯',
 	u'我x':u'我操',
 	u'x你妈':u'操你妈',
 	u'p用':u'屁用',
 	u'胎毒':u'台独',
 	u'DT':u'蛋疼',
 	u'dt':u'蛋疼',
 	u'IT':u'信息技术',
 	u'1楼':u'一楼',
 	u'2楼':u'二楼',
 	u'2逼':u'二逼',
 	u'二b':u'二逼',
 	u'二B':u'二逼',
 	u'晚9':u'晚九',
 	u'朝5':u'朝五',
 	u'黄易':u'黄色网易',
 	u'艹':u'操',
 	u'滚下抬':u'滚下台',
 	u'灵道':u'领导',
 	u'煳':u'糊',
 	u'跟贴被火星网友带走啦':u'',
 	u'棺猿':u'官员',
 	u'贯猿':u'官员',
 	u'巢县':u'朝鲜',
 	u'死大林':u'斯大林',
 	u'无毛们':u'五毛们',
 	u'天巢':u'天朝',
 	u'普特勒':u'普京',
 	u'依拉克':u'伊拉克',
 	u'歼20':u'歼二零',
 	u'歼10':u'歼十',
 	u'歼8':u'歼八',
 	u'f22':u'猛禽',
 	u'p民':u'屁民',
 	u'钟殃':u'中央',
    u'３':u'三',
    "１": "一",
    u'２':u'二',
    u'４':u'四',
    u'５':u'五',
    u'６':u'六',
    u'７':u'七',
    u'８':u'八',
    u'９':u'九',
    u'０':u'零'
	}
    
    #jieba.load_userdict("stop_dict/tang_dict.txt")
    text = re.sub('（.*?）', '', text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，？、~@#￥%……&*（）]+", "",text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', '', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', text)
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\”', '',text)
    text = re.sub(r'\＃', '', text)
    text = re.sub(r'\“', '',text)
    text = re.sub(r'\《', '',text)
    text = re.sub(r'\》', '',text)
    text = re.sub(r'\［', '',text)
    text = re.sub(r'\］', '', text)
    text = re.sub(r'\“', '', text)
    text = re.sub(r'\＂', '', text)
    text = re.sub(r'\ue40c0', '', text)
    
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = "".join(new_text)
    
    text = jieba.cut(text)
    text = " ".join(text)
    text1 = text
    
#     stopw = [line.strip() for line in open('stop_dict/stops.txt', encoding= 'utf-8').readlines()]
#     if remove_stopwords:
#         text1 = ''
#         for word in text: 
           
#             if word not in stopw:
#                 if word != '\t':
#                     word = re.sub(r'  ', ' ', word)
#                     text1 += word
#     text1 = re.sub(r'  ', ' ', text1)
    return text1

# # pickle
# 按顺序读取    存放顺序为 int_to_vocab  vocab_to_int word_embedding_matrix

# init
def init():
    f2 = open('parameter.pkl', 'rb')
    int_to_vocab = pickle.load(f2)
    vocab_to_int = pickle.load(f2)
    word_embedding_matrix = pickle.load(f2)
    f2.close()
    checkpoint = "./Model/best_model_24.ckpt"
    batch_size = 32
    return int_to_vocab, vocab_to_int,word_embedding_matrix, batch_size,checkpoint


def text_to_seq(text,vocab_to_int):
    '''Prepare the text for the model'''
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


def test(input_sentence, text,checkpoint, int_to_vocab, vocab_to_int, word_embedding_matrix, batch_size):

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                          summary_length: [np.random.randint(5,20)], 
                                          text_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0] 

    pad = vocab_to_int["<PAD>"] 

    print('Original Text:', input_sentence)

    print('\nText')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
    return " ".join([int_to_vocab[i] for i in answer_logits if i != pad])





from app import App

class Textsum(App):
    def __init__(self):
        super().__init__()

    def OnInitProgrmne(self):
        self.int_to_vocab, self.vocab_to_int, self.word_embedding_matrix, self.batch_size, self.checkpoint=init()

    def OnTextSum(self, text_):
        text = text_to_seq(text_,self.vocab_to_int)
        return test(text_, text,self.checkpoint, self.int_to_vocab, self.vocab_to_int, self.word_embedding_matrix, self.batch_size)




if __name__=="__main__":
	Textsum().MainLoop()
