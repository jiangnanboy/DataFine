import onnxruntime as rt
import os

class AbsModel:
    def __init__(self, det_model_path):
        print('load model...')
        if not os.path.exists(det_model_path):
            raise ValueError("not find model file path {}".format(
                det_model_path))
        self.load_det_model(det_model_path)

    def load_det_model(self, det_model_path):
        session_options = rt.SessionOptions()
        session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 2
        self.sess = rt.InferenceSession(det_model_path, session_options)