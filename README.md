模型图：项目中model.png  
请参照模型图理解代码  

1.项目大致流程描述：  
word/char Embedding(特征嵌入层):  
在词级别的向量基础上加入字符级的信息，这样的embedding可以捕捉前缀后缀这样的形态特征。  
先用skip-gram word2vec 模型预训练得到的词向量表将每个词映射为一个词向量，然后把每个词中字母用一个向量表示，把一个词中所包含的字母的向量送入 BiLSTM,
把前后两个最终状态和 词向量进行拼接,得到词的embedding  
BiLSTM层:  
把句子中所包含词的embedding输入，然后将前向、后向 每个对应位置的hidden state拼接起来得到新的编码序列。  
CRF Layer:    
采用BIO标注策略,使用CRF引入标签之间的依赖关系，  
计算每个词得到不同标签的分数  
计算句子的标签序列概率  
采用Viterbi算法得到分数最高的序列标签  
在进行命名实体时 通过最小化交叉熵损失 来达到 优化网络参数和CRF的目的，测试时用Viterbi算法得到分数最高的序列标签  
Label Embedding:  
实体标签的embedding。训练时真实标签,测试时为预测标签  
Heads Relations：    
输入为BiLSTM的hidden state和label Embedding的拼接。可以预测多个头，头和关系的决策是一块完成的,而不是先预测头，再用关系分类器预测关系  
标签策略： CRF层的输出是采用BIO标注策略的实体识别结果，head Relations层只有在和其他实体有关系时  会给出对应实体的尾单词和关系；在与其他实体没有关系时 head为原单词本身,关系为N  
Adversarial training(AT):  对抗训练 使分类器对于噪音数据有更强的鲁棒性(混合原来的样本+对抗样本)  


词向量数据路径：  
链接: https://pan.baidu.com/s/1P_QtMKKhUdtc0XfOnpSBOw 提取码: 45ic 

2.数据格式描述：  
#doc 5121  文件名  
['token_id', 'token', "BIO", "relation", 'head']  
token_id : 每个文件中词所在位置下标  
token :    词  
BIO：      标注实体类型  
relation:  实体关系  
head:      当前 实体关系 对应实体的位置下标  

data_parsers.py:  
docId:        文件名称id  
token_ids:    词在每个文件中对应位置的下标列表  
tokens:       单词的列表  
BIOs:         词对应的实体列表  
ecs:          没加标注的的实体列表  
relations:    实体关系的列表  
heads:        实体关系对应实体下标位置的列表，如[[2],[3,4]]  
char_ids:     每个单词中的每个字母对应的id的列表,如 两个单词第一个单词包含三个字母,第二个单词包含四个字母[[1,2,3],[11,12,1,4]]  
embedding_ids:单词对应id的列表  
BIO_ids:      实体对应id的列表  
ec_ids:       没加标注的实体对应id的列表  
joint_ids:    实体关系联合的列表：计算规则(可参考后期验证数据校验时的 数据处理规则)：headId*len(set(relations))+relation_id  
实体关系的去重列表长度:len(set(relations))  
该实体谷关系对应的实体下标:headId  
实体关系 对应的id: relation_id  



3.文件描述：方法详细功能在代码注释中可看  
data_build.py    初始化配置文件及数据  
data_parsers.py  封装数据  
model.py         模型  
train.py         模型训练  
data_utils       数据转换、处理  
eval             模型校验  
