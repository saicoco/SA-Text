# coding=utf-8
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape=[1,3,512,512]):

    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file. 
        shape : Shape of the input of the ONNX file. 
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)
        
def load_engine(trt_runtime, plan_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
    
if __name__=='__main__':
    from onnx import ModelProto
    import sys
    ONNX_MODEL = sys.argv[1]
    out_model = sys.argv[2]
    model = ModelProto()
    with open(ONNX_MODEL, "rb") as f:
        model.ParseFromString(f.read())
    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [1, d0, d1 ,d2]
    engine = build_engine(ONNX_MODEL)
    save_engine(engine, out_model)