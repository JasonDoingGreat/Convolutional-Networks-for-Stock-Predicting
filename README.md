# Convolutional Networks for Stock Predicting

This is a Machine Learning project using Convolutional Neural Network to predict the stock price. It is done after a project article by Ashwin Siripurapu from Stanford University but in a different way.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to run the program normally, please make sure you have a Python compiler, such as Anaconda, PyCharm or IDLE.

Next, you also need some dependencies:

```
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
```

Also, in this program, we are using GPU to accelerate the running speed.

```
Accepted NVIDIA GPU for acceleration:
	Check out the official website: 
		https://developer.nvidia.com/cuda-gpus
Very encourage to run this program on anaconda on Windows.
Implement details for Keras on Windows using GPU: 
	https://datanoord.com/2016/02/02/setup-a-deep-learning-environment-on-windows-theano-keras-with-gpu-enabled/
Other details please refer to the report.
```

### Usage

```
Check the Baseline part:
	After installing all the libraries required, simply run the main.py file should work.
	If not working, then modify all the paths appeared in the main.py to your working directory.

Check the CNN part:
	Simply run the cnn_main.py should work.
	If not, then modify all the paths appeared in the file like the previous one.
```
All the generated images will be stored in 'figures' and 'figures_v2' files seperately. Please make sure you have these two files in the project file. The running time depends on the GPU.


## Authors

* **Zezhou Li** - *Modified work of initial work by Ashwin Siripurapu* - See Acknowledgments

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **Ashwin Siripurapu** - *Initial work* - Paper at Stanford University
*	http://cs231n.stanford.edu/reports/2015/pdfs/ashwin_final_paper.pdf
