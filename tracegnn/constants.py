import os

# if MIN_NODE_COUNT <= 2 <= MAX_NODE_COUNT, then the graph will be chosen
MAX_NODE_COUNT = int(os.environ.get('MAX_NODE_COUNT', '32'))
MAX_SPAN_COUNT = int(os.environ.get('MAX_SPAN_COUNT', '32'))

# whether or not to use multi-dimensional latency codec?
# If not set, will normalize the latency w.r.t. each operation.
USE_MULTI_DIM_LATENCY_CODEC = os.environ.get('USE_MULTI_DIM_LATENCY_CODEC', '0') == '1'

# If USE_MULTI_DIM_LATENCY_CODEC, then encode the codec parameters.
MAX_LATENCY_DIM = int(os.environ.get('MAX_LATENCY_DIM', '5'))
MAX_DEPTH = int(os.environ.get('MAX_DEPTH', '4'))
