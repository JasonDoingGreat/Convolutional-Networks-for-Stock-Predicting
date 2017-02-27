<snippet>
  <content>
# Convolutional Networks for Stock Predicting

This project aims to train a convolutional networks for stock price predicting.
Using Python as the programming language.
Using Anaconda on mac or Windows.
Using GPU to accelerate computation.
Using Keras Library for CNN model.
There are mainly two parts:
	1. Baseline: Implementing OLS model to do a quick test.
	2. CNN model: Implementing CNN model for stock price images and do a non-linear test.
Two python files:
	1. main.py
	2. cnn_main.py

## Requirements

Libraries:
	Baseline: main.py
		1. numpy
		2. matplotlib
		3. glob
		4. math
		5. PIL
		6. statsmodels

	Convolutional Networks: cnn_main.py
		1. numpy
		*2. keras	
		3. scipy
		4. glob
		5. matplotlib
		6. PIL
		7. math

	GPU acceleraton:
		Accepted NVIDIA GPU for acceleration:
			Check out the official website: 
				https://developer.nvidia.com/cuda-gpus
		Very encourage to run this program on anaconda on Windows.
		Implement details for Keras on Windows using GPU: 
			https://datanoord.com/2016/02/02/setup-a-deep-learning-environment-on-windows-theano-keras-with-gpu-enabled/
		Other details please refer to the report.	

## Usage

Check the Baseline part:
	After installing all the libraries required, simply run the main.py file should work.
	If not working, then modify all the paths appeared in the main.py to your working directory.

Check the CNN part:
	Simply run the cnn_main.py should work.
	If not, then modify all the paths appeared in the file like the previous one.

All the generated images will be stored in 'figures' and 'figures_v2' files seperately. The running time depends on the GPU.

## License

MIT License
</content>
</snippet>
