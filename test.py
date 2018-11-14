import tensorflow as tf
import numpy as np

# 模型目录
CHECKPOINT_DIR = './model'
INCEPTION_MODEL_FILE = './inception_dec_2015/tensorflow_inception_graph.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称
flower_dict = {0:'sunflowers',1:'dandelion',2:'dasiy',3:'roses',4:'tulips'}
# 测试数据
#file_path = './flower_photos/tulips/11746080_963537acdc.jpg'
#file_path = "./flower_photos/daisy/5547758_eea9edfd54_n.jpg"
#file_path = "./flower_photos/dandelion/7355522_b66e5d3078_m.jpg"
#file_path = "./flower_photos/roses/394990940_7af082cf8d_n.jpg"
#file_path = "./flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
file_path = "./flower_photos/tulips/10791227_7168491604.jpg"



# 读取数据
image_data = tf.gfile.FastGFile(file_path, 'rb').read()

# 评估
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        # 读取训练好的inception-v3模型
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 使用inception-v3处理图片获取特征向量
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {jpeg_data_tensor: image_data})
        # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
        bottleneck_values = [np.squeeze(bottleneck_values)]
        print(checkpoint_file)
        # 加载元图和变量
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 通过名字从图中获取输入占位符
        input_x = graph.get_operation_by_name(
            'BottleneckInputPlaceholder').outputs[0]

        # 我们想要评估的tensors
        predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

        # 收集预测值
        all_predictions = []
        all_predictions = sess.run(predictions, {input_x: bottleneck_values})
print(flower_dict[all_predictions[0]])


