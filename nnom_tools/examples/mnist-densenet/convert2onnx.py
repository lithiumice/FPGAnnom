import tensorflow.keras
import onnx
from tensorflow.keras.models import load_model, save_model

model = load_model('mnist_model.h5')

temp_model_file = 'mnist_onnx.onnx'
onnx.save_model(onnx_model, temp_model_file)

save_model(model, 'mnist_tf',save_format='tf')


from tensorflow.keras.models import load_model
import tensorflow as tf
import os
# from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_util, graph_io
import tensorflow as tf
from tensorflow.python.platform import gfile

def h5_to_pb(h5_weight_path, output_dir, out_prefix="output_", log_tensorboard=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    h5_model =  load_model(h5_weight_path)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    model_name = os.path.splitext(os.path.split(h5_weight_path)[-1])[0] + '.pb'

    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

#读取模型各层
def read_pb(GRAPH_PB_PATH):
    with tf.Session() as sess:
        print("load graph！！！")
        with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            for i, n in enumerate(graph_def.node):
                print("Name of the node - %s" % n.name)
if __name__ == '__main__':
    h5_to_pb(h5_weight_path='mnist_model.h5', output_dir='./')
    # read_pb('./model.pb')