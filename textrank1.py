
# coding: utf-8

# In[1]:

import math  
from string import punctuation  
from heapq import nlargest  
from itertools import product, count   
import numpy as np 
import pandas as pd
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
import jieba
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from gensim.models import Word2Vec
import pprint, pickle
from gui.app import Start_GUI
from gui.app import TextSumApi
# In[4]:

model = Word2Vec.load('wordembedding/wiki.zh.text.model')




# In[9]:

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

def clean_text1(text, remove_stopwords = True):
    
    text = re.sub('（.*?）', '', text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！？、~@#￥%……&*（）]+", "",text)
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
    text = re.sub(r'\］', '',text)
    text = re.sub(r'\“', '', text)
    text = re.sub(r'\＂', '', text)
    
    #jieba.load_userdict("dict.txt")
    text = jieba.cut(text)
    text = " ".join(text)
    
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = "".join(new_text)
    text = re.sub(r' ', '', text)
    text = re.sub(r'\ue40c0', '', text)
    
    

    return text

# 初始化
def init_rank():
    f2 = open('parameter_textrank.pkl', 'rb')
    vocab_to_int = pickle.load(f2)
    word_embedding_matrix = pickle.load(f2)
    f2.close()
    checkpoint = "./Model/best_model_24.ckpt"
    batch_size = 32
    return vocab_to_int,word_embedding_matrix, batch_size,checkpoint



def two_sentences_similarity(sents_1, sents_2):  
    ''''' 
    计算两个句子的相似性
    :param sents_1: 
    :param sents_2: 
    :return: 
    '''  
    counter = 0  
    for sent in sents_1:  
        if sent in sents_2:  
            counter += 1  
    return counter / (math.log(len(sents_1) + len(sents_2))) 



def create_graph(word_sent,vocab_to_int,word_embedding_matrix):
    """ 
    传入句子链表  返回句子之间相似度的图 
    :param word_sent: 
    :return: 
    """ 
    num = 0
    num = len(word_sent)  
    board = [[0.0 for _ in range(num)] for _ in range(num)]  
  
    for i, j in product(range(num), repeat=2):  
        if i != j:  
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j],vocab_to_int,word_embedding_matrix)
    return board



def cosine_similarity(vec1, vec2):  
    ''''' 
    计算两个向量之间的余弦相似度 
    :param vec1: 
    :param vec2: 
    :return: 
    '''  
    tx = np.array(vec1)  
    ty = np.array(vec2)  
    cos1 = np.sum(tx * ty)  
    cos21 = np.sqrt(sum(tx ** 2))  
    cos22 = np.sqrt(sum(ty ** 2))  
    cosine_value = cos1 / float(cos21 * cos22)  
    return cosine_value



def compute_similarity_by_avg(sents_1, sents_2,vocab_to_int,word_embedding_matrix):
    ''''' 
    对两个句子求平均词向量 
    :param sents_1: 
    :param sents_2: 
    :return: 
    '''  
    if len(sents_1) == 0 or len(sents_2) == 0:  
        return 0.0 
    

    x = sents_1[0]
    vec1 = word_embedding_matrix[vocab_to_int[x]]  
    
    for word1 in sents_1[1:]:
        intid1 = vocab_to_int[word1]
        vec1 = vec1 + word_embedding_matrix[intid1]  
  
    vec2 = word_embedding_matrix[vocab_to_int[sents_2[0]]] 
    for word2 in sents_2[1:]:
        intid2 =  vocab_to_int[word2]
        vec2 = vec2 + word_embedding_matrix[intid2]  
  
    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))  
    return similarity  



def calculate_score(weight_graph, scores, i):  
    """ 
    计算句子在图中的分数 
    :param weight_graph: 
    :param scores: 
    :param i: 
    :return: 
    """  
    length = len(weight_graph)  
    d = 0.85  
    added_score = 0.0  
  
    for j in range(length):  
        fraction = 0.0  
        denominator = 0.0  
        # 计算分子  
        fraction = weight_graph[j][i] * scores[j]  
        # 计算分母  
        for k in range(length):  
            denominator += weight_graph[j][k]  
            if denominator == 0:  
                denominator = 1  
        added_score += fraction / denominator  
    # 算出最终的分数  
    weighted_score = (1 - d) + d * added_score  
    return weighted_score  



def weight_sentences_rank(weight_graph):  
    ''''' 
    输入相似度的图（矩阵) 
    返回各个句子的分数 
    :param weight_graph: 
    :return: 
    '''  
    # 初始分数设置为0.5  
    scores = [0.5 for _ in range(len(weight_graph))]  
    old_scores = [0.0 for _ in range(len(weight_graph))]  
  
    # 开始迭代  
    while different(scores, old_scores):  
        for i in range(len(weight_graph)):  
            old_scores[i] = scores[i]  
        for i in range(len(weight_graph)):  
            scores[i] = calculate_score(weight_graph, scores, i)  
    return scores  



def different(scores, old_scores):  
    ''''' 
    判断前后分数有无变化 
    :param scores: 
    :param old_scores: 
    :return: 
    '''  
    flag = False  
    for i in range(len(scores)):  
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:  
            flag = True  
            break  
    return flag 




def filter_symbols(sents):  
    stopwords = create_stopwords() + ['。', ' ', '.']  
    _sents = []  
    for sentence in sents:  
        for word in sentence:  
            if word in stopwords:  
                sentence.remove(word)  
        if sentence:  
            _sents.append(sentence)  
    return _sents  


def filter_model(sents):  
    _sents = []  
    for sentence in sents:  
        for word in sentence:  
            if word not in model:  
                sentence.remove(word)  
        if sentence:  
            _sents.append(sentence)  
    return _sents  



def cut_sentences(sentence):  
    puns = frozenset(u'。！？')  
    tmp = []  
    for ch in sentence:  
        tmp.append(ch)  
        if puns.__contains__(ch):  
            yield ''.join(tmp)  
            tmp = []  
    yield ''.join(tmp)


# In[29]:

# 句子中的stopwords  
def create_stopwords():  
    stop_list = [line.strip() for line in open("stopwords.txt", 'r', encoding='utf-8').readlines()]  
    return stop_list  


# In[30]:

def summarize0(text, n,word_embedding_matrix,vocab_to_int):
    tokens = cut_sentences(text) 
    sentences = []  
    sents = []
    list1 = []
    for sent in tokens: 
        sentences.append(sent)  
        sents.append([word for word in jieba.cut(sent) if word])  
  
    # sents = filter_symbols(sents)  
    sents = filter_model(sents)  
    graph = create_graph(sents,vocab_to_int,word_embedding_matrix)

    scores = weight_sentences_rank(graph)
    sent_selected = nlargest(n, zip(scores,sentences))
    
    sent_index = [] 

    for i in range(n):
        try:
            sent_index.append(sent_selected[i][1])
        except:
            sent_index.append('')
    return sent_index  


# In[ ]:

rank_texts = []
text = '中广网唐山６月１２日消息 据中国之声《新闻晚高峰》报道，今天（日）上午，公安机关年缉枪制爆专项行动“统一销毁非法枪爆物品活动”在河北唐山正式启动，１０万余只非法枪支、５０余吨炸药在全国１５０个城市被统一销毁。黄明：现在我宣布，全国缉枪制爆统一销毁行动开始！随着公安部副部长黄明一声令下，大量仿制式枪以及猎枪、火药枪、气枪在河北唐山钢铁厂被投入炼钢炉。与此同时，在全国各省区市１５０个城市，破案追缴和群众主动上缴的１０万余支非法枪支被集中销毁，在全国各指定场所，２５０余吨炸药被分别销毁。公安部治安局局长刘绍武介绍，这次销毁的非法枪支来源于三个方面。刘绍武：打击破案包括涉黑、涉恶的团伙犯罪、毒品犯罪，还有从境外非法走私的枪支爆炸物。在销毁现场，记者看到了被追缴和上缴的各式各样的枪支。刘绍武：也包括制式枪，有的是军用枪、仿制的制式抢，还有猎枪、私制的火药枪等等。按照我国的枪支管理法，这些都是严厉禁止个人非法持有的。中国是世界上持枪犯罪的犯罪率最低的国家之一。中美联手破获特大跨国走私武器弹药案近日，中美执法部门联手成功破获特大跨国走私武器弹药案，在中国抓获犯罪嫌疑人２３名，缴获各类枪支９３支、子弹５万余发及大量枪支配件。在美国抓获犯罪嫌疑人３名，缴获各类枪支１２支。这是公安部与美国移民海关执法局通过联合调查方式侦破重大跨国案件的又一成功案例。０１１年８月５日，上海浦东国际机场海关在对美国纽约发往浙江台州，申报品名为扩音器（音箱）的快件进行查验时，发现货物内藏有手枪９支，枪支配件９件，长枪部件７件。经检验，这些都是具有杀伤力的制式枪支及其配件。这引起了公安部和海关总署的高度重视。公安部刑侦局局长刘安成：因为是从海关进口的货物中检查出来夹带，说明来源地是境外，或是说国外，这应该是一起特大跨国走私武器弹药的案件。上海市公安局和上海海关缉私局成立联合专案组，迅速开展案件侦查。专案组于８月２６日在浙江台州ＵＰＳ取件处将犯罪嫌疑人王挺（男，３２岁，台州市人）抓获。王挺交代，他通过一境外网站上认识了上家林志富，２００９年１１月以来，林志富长期居住美国，他通过互联网组建了一个走私、贩卖、私藏枪支弹药的群体，通过网络在国内寻找枪支弹药买家，并通过美国ＵＰＳ联邦速递公司将枪支弹药从纽约快递给多名类似王挺的中间人，再通过中间人发送给国内买家。此案中，犯罪分子依托虚拟网络进行犯罪交易，隐蔽性强，涉案人员使用的身份、地址、联系方式都是虚构的，侦查难度很大。刘安成说，此案体现了是新型犯罪，特别是现代犯罪的新特点。刘安成：他不受距离的限制、经常是跨国跨境，甚至是跨一个、数个、甚至数十个国家。这种犯罪手法的改变和新型犯罪的特点，要求我们各国警方充分合作。作者：汤一亮庄胜春'
text = clean_text1(text)
text.replace('\n', '')



#text_rank=summarize0(text,int(1),word_embedding_matrix1,vocab_to_int1)

# 摘取式
#print(text_rank)
#print(text_rank[0])


def clean_text(text, remove_stopwords=True):

    # jieba.load_userdict("stop_dict/tang_dict.txt")
    text = re.sub('（.*?）', '', text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，？、~@#￥%……&*（）]+", "", text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', '', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', text)
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\”', '', text)
    text = re.sub(r'\＃', '', text)
    text = re.sub(r'\“', '', text)
    text = re.sub(r'\《', '', text)
    text = re.sub(r'\》', '', text)
    text = re.sub(r'\［', '', text)
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
    return int_to_vocab, vocab_to_int, word_embedding_matrix, batch_size, checkpoint


def text_to_seq(text, vocab_to_int):
    '''Prepare the text for the model'''
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


def test(input_sentence, text, int_to_vocab, vocab_to_int, word_embedding_matrix, batch_size,checkpoint):
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

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          summary_length: [np.random.randint(5, 20)],
                                          text_length: [len(text)] * batch_size,
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


#int_to_vocab, vocab_to_int, word_embedding_matrix, batch_size, checkpoint = init()
#text = text_to_seq(text_rank[0],vocab_to_int)
#生成式
#textsum = test(text_rank[0], text,checkpoint, int_to_vocab, vocab_to_int, word_embedding_matrix, batch_size)



import time


class api(TextSumApi):
    #this function return the tensorbox website url
    def get_html_url(self):
        return "http://news.163.com/18/0123/19/D8S08G4K000189FH.html"
    #this function init the program
    def OnInit(self):
        self.init_arg = init_rank()
        self.init_plus = init()
        pass



    #this function do the 摘取式文本摘要,return (文本摘要,rouge)
    def OnSelTextSum(self,text):
        text = clean_text1(text)
        text.replace('\n', '')
        self.sel = summarize0(text, int(1), self.init_arg[1],self.init_arg[0])
        return (self.sel[0],0)


    # this function do the 生成式文本摘要,return (文本摘要,rouge)
    def OnGenTextSum(self,text):
        text = text_to_seq(self.sel[0],self.init_plus[1])
        return (test(self.sel[0],text,*self.init_plus),0)

    #this function return the help information string（html format)
    def get_help_url(self):
        return "<hrml>hello</html>"
        pass

Start_GUI(api())
