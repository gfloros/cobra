"""Common configuration parameters for the project."""

DATA_PATH = './data/'
MODELS_PATH = './data/models/'
SAMPLED_POINTS_PATH = './data/sampled_points/'
GP_WEIGHTS_PATH = './data/gp_weights/'
RESULTS_PATH = './data/results/'
DATASETS_PATH = './data/datasets/'
SCORING_PATH = './scoring/'
DEBUG = False

SCALING_FACTOR = 1.03
TRAIN_RANDOM_SEED = 42
TEST_RANDOM_SEED = 102
NUM_SAMPLES = [10000, 250000]
NUM_CENTERS = 100
VCAMERA_FOV = 90  # degrees
VCAMERA_WIDTH = 640  # pixels
VCAMERA_HEIGHT = 480  # pixels
VCAMERA_UP_VECTOR = [0, 1, 0]  # Assuming y-axis is up; adjust as necessary
SCENE_CENTER = [0, 0, 0]  # Look at the center of the model
