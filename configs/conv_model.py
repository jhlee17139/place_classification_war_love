from easydict import EasyDict as edict


config = edict()
config.network = "conv"

config.epochs = 80
config.batch = 512
config.learning_rate = 0.01
config.resize = 128
config.random_rotate = 10.0
config.device = "cuda:0"

config.weight_path = "../love_war_place/weight"
config.train_data_path = "../love_war_place/data/train"
config.val_data_path = "../love_war_place/data/val"
config.inference_input = "../love_war_place/inference/input"
config.inference_output = "../love_war_place/inference/output"

config.labels = ['car', 'front_of_buliding', 'hospital', 'house', 'indoor', 'restaurant', 'rooftop', 'street']


