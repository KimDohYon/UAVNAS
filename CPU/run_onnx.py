import numpy as np
import onnxruntime as ort
import time


input_data = np.fromfile("train06_trimmed.fc32", dtype=np.float32).reshape(1, 2, 32, 3072)


session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])


input_name = session.get_inputs()[0].name


start = time.time()
output = session.run(None, {input_name: input_data})
end = time.time()

print("CPU inference time: {:.3f} ms".format((end - start) * 1000))
