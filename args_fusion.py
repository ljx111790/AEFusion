
class args():
	# training args
	epochs = 2  # "number of training epochs, default is 2"
	batch_size = 4 # "batch size for training, default is 4"
	# the COCO dataset path in your computer
	# URL: http://images.cocodataset.org/zips/train2014.zip
	dataset = "E:/dataset/train2014/"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "models/AEFusion_autoencoder"
	Final_model_dir = "models"
	save_train_loss_or_not = False
	save_Final_loss_or_not = False
	save_loss_dir = './models/AEFusion_mobel/'



	cuda = 1
	ssim_weight = [1, 10, 100, 1000, 10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4
	lr_light = 1e-4
	log_interval = 10
	resume = None

	# for test
	model_default = './models/aefusion.model'  # test model

	#model_deepsuper = './models/Final_epoch_2.model'


