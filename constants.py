datasets = ('left_240x20_fixed', 'right_240x20_fixed',
            'left_240x40_fixed', 'right_240x40_fixed',
            'left_240x20', 'right_240x20',
            'left_240x40', 'right_240x40')
# datasets = ('left_240x20_fixed', 'right_240x20_fixed',
#             'left_240x40_fixed', 'right_240x40_fixed',
#             'left_240x20', 'right_240x20', 'left_240x40',
#             'right_240x40', 'left_480x80_fixed', 'right_480x80_fixed',
#             'left_480x80', 'right_480x80')
MASK_VALUE = 0
TEST_SIZE = 0.3
SCALE_DATASET = True
N_PARTS = 10  # Number of partitions that are being used in total
PAIR_METHOD = 'agrowth_hung_10'
PARAMS_PARTITION = 1
MODEL_PARAMS_FOLDER = 'model_params'
ROOT_DATA_FOLDER = 'data'

LBP_METHOD = 'uniform'
LBP_RADIUS = 3
LBP_NPOINTS = 8 * LBP_RADIUS
