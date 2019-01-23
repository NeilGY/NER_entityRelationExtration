import data_utils
import model as Model
from data_build import data_build
import numpy as np
import tensorflow as tf


output_dir='logs/'
config_file='config/bio_config'

def train():
    config = data_build(config_file) #加载配置文件数据,处理训练数据
    train_data = data_utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))
    dev_data = data_utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))
    test_data = data_utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))

    tf.reset_default_graph()
    tf.set_random_seed(1)

    data_utils.printParameters(config)

    with tf.Session() as sess:
        embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,
                                           trainable=False).assign(config.wordvectors)
        emb_mtx = sess.run(embedding_matrix)
        #初始化模型
        model = Model.model(config, emb_mtx, sess)
        #获取需要计算的模型损失、预测结果
        obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel = model.run()
        #优化函数迭代
        train_step = model.get_train_op(obj)
        #模型参数
        operations = Model.operations(train_step, obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel)

        sess.run(tf.global_variables_initializer())

        best_score = 0
        nepoch_no_imprv = 0  # for early stopping

        for iter in range(config.nepochs + 1):
            #模型训练
            model.train(train_data, operations, iter)
            #模型评估
            dev_score = model.evaluate(dev_data, operations, 'dev')
            model.evaluate(test_data, operations, 'test')

            if dev_score >= best_score:
                nepoch_no_imprv = 0
                best_score = dev_score

                print("- Best dev score {} so far in {} epoch".format(dev_score, iter))

            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= config.nepoch_no_imprv:
                    print("- early stopping {} epochs without " \
                          "improvement".format(nepoch_no_imprv))

                    with open(output_dir + "/es" + ".txt", "w+") as myfile:
                        myfile.write(str(iter))
                        myfile.close()

                    break

def main(_):
    train()
if __name__ == '__main__':
    tf.app.run(main)