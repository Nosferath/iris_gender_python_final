datasets = ('left_240x20_fixed', 'right_240x20_fixed',
            'left_240x40_fixed', 'right_240x40_fixed',
            'left_240x20', 'right_240x20',
            'left_240x40', 'right_240x40')
# datasets = ('left_240x20_fixed', 'right_240x20_fixed',
#             'left_240x40_fixed', 'right_240x40_fixed',
#             'left_240x20', 'right_240x20', 'left_240x40',
#             'right_240x40', 'left_480x80_fixed', 'right_480x80_fixed',
#             'left_480x80', 'right_480x80')
datasets_botheyes = ('240x20_fixed', '240x40_fixed', '240x20', '240x40')
MASK_VALUE = 0
TEST_SIZE = 0.2
SCALE_DATASET = True
N_PARTS = 10  # Number of partitions that are being used in total
PAIR_METHOD = 'agrowth_hung_10'
PARAMS_PARTITION = 1
MODEL_PARAMS_FOLDER = 'model_params'
ROOT_DATA_FOLDER = 'data'
ROOT_PERI_FOLDER = 'data_peri'
SPP_FOLDER = 'spp_mats'

FEMALES_LABEL = 0
MALES_LABEL = 1

PERIOCULAR_SHAPE = (480, 640)

LBP_METHOD = 'uniform'
LBP_RADIUS = 3
LBP_NPOINTS = 8 * LBP_RADIUS
