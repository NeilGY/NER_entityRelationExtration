import os
import data_utils
import data_parsers as parsers
from sklearn.externals import joblib
import os.path

""""Read the configuration file and set the parameters of the model"""


class data_build():
    def __init__(self, fname):

        config_file = parsers.read_properties(fname) #加载配置文件
        self.config_fname = fname

        # load data
        self.filename_embeddings = config_file.getProperty("filename_embeddings")
        self.filename_train = config_file.getProperty("filename_train")
        self.filename_test = config_file.getProperty("filename_test")
        self.filename_dev = config_file.getProperty("filename_dev")
        #生成各列数据的集合
        self.train_id_docs = parsers.readHeadFile(self.filename_train)
        self.dev_id_docs = parsers.readHeadFile(self.filename_dev)
        self.test_id_docs = parsers.readHeadFile(self.filename_test)

        # 将所有数据加到一个大集合中
        dataset_documents = []
        dataset_documents.extend(self.train_id_docs)
        dataset_documents.extend(self.dev_id_docs)
        dataset_documents.extend(self.test_id_docs)

        self.dataset_set_characters = data_utils.getCharsFromDocuments(dataset_documents)#获得所有数据中 字母 数字的集合
        self.dataset_set_bio_tags, self.dataset_set_ec_tags = data_utils.getEntitiesFromDocuments(dataset_documents)#获得所有数据中 实体 的集合
        self.dataset_set_relations = data_utils.getRelationsFromDocuments(dataset_documents)#获得所有数据中 关系 的集合
        #加载预训练好的词向量
        if os.path.isfile(self.filename_embeddings + ".pkl") == False:
            self.wordvectors, self.representationsize, self.words = data_utils.readWordvectorsNumpy(self.filename_embeddings, isBinary=True if self.filename_embeddings.endswith(".bin") else False)
            self.wordindices = data_utils.readIndices(self.filename_embeddings,
                                                 isBinary=True if self.filename_embeddings.endswith(".bin") else False)
            joblib.dump((self.wordvectors, self.representationsize, self.words, self.wordindices), self.filename_embeddings + ".pkl")

        else:
            self.wordvectors, self.representationsize, self.words, self.wordindices = joblib.load(self.filename_embeddings + ".pkl")  # loading is faster
        #将数据转换成对应id的列表
        parsers.preprocess(self.train_id_docs, self.wordindices, self.dataset_set_characters,
                           self.dataset_set_bio_tags, self.dataset_set_ec_tags, self.dataset_set_relations)
        parsers.preprocess(self.dev_id_docs, self.wordindices, self.dataset_set_characters,
                           self.dataset_set_bio_tags, self.dataset_set_ec_tags, self.dataset_set_relations)
        parsers.preprocess(self.test_id_docs, self.wordindices, self.dataset_set_characters,
                           self.dataset_set_bio_tags, self.dataset_set_ec_tags, self.dataset_set_relations)

        # training
        self.nepochs = int(config_file.getProperty("nepochs"))
        self.optimizer = config_file.getProperty("optimizer")
        self.activation = config_file.getProperty("activation")
        self.learning_rate = float(config_file.getProperty("learning_rate"))
        self.gradientClipping = data_utils.strToBool(config_file.getProperty("gradientClipping"))
        self.nepoch_no_imprv = int(config_file.getProperty("nepoch_no_imprv"))
        self.use_dropout = data_utils.strToBool(config_file.getProperty("use_dropout"))
        self.ner_loss = config_file.getProperty("ner_loss")
        self.ner_classes = config_file.getProperty("ner_classes")
        self.use_chars = data_utils.strToBool(config_file.getProperty("use_chars"))
        self.use_adversarial = data_utils.strToBool(config_file.getProperty("use_adversarial"))

        # hyperparameters
        self.dropout_embedding = float(config_file.getProperty("dropout_embedding"))
        self.dropout_lstm = float(config_file.getProperty("dropout_lstm"))
        self.dropout_lstm_output = float(config_file.getProperty("dropout_lstm_output"))
        self.dropout_fcl_ner = float(config_file.getProperty("dropout_fcl_ner"))
        self.dropout_fcl_rel = float(config_file.getProperty("dropout_fcl_rel"))
        self.hidden_size_lstm = int(config_file.getProperty("hidden_size_lstm"))
        self.hidden_size_n1 = int(config_file.getProperty("hidden_size_n1"))
        # self.hidden_size_n2 = config_file.getProperty("hidden_size_n2")
        self.num_lstm_layers = int(config_file.getProperty("num_lstm_layers"))
        self.char_embeddings_size = int(config_file.getProperty("char_embeddings_size"))
        self.hidden_size_char = int(config_file.getProperty("hidden_size_char"))
        self.label_embeddings_size = int(config_file.getProperty("label_embeddings_size"))
        self.alpha = float(config_file.getProperty("alpha"))

        # evaluation
        self.evaluation_method = config_file.getProperty("evaluation_method")
        self.root_node = data_utils.strToBool(config_file.getProperty("root_node"))

        self.shuffle = False
        self.batchsize = 16






