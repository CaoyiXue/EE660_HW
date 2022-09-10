import numpy as np
import matplotlib.pyplot as plt

def MSE(X,Y,w):
	return np.mean(np.square(np.dot(X,w)-Y))

def display(w,Xtest,Ytest,ax,norm='L2',
	levels=None,
	w1_range=(-4.0, 6.1, 100),
	w2_range=(-4.0, 6.1, 100)):

	w = np.array(w)

	w1list = np.linspace(w1_range[0], w1_range[1], w1_range[2])
	w2list = np.linspace(w2_range[0], w2_range[1], w2_range[2])
	W1, W2 = np.meshgrid(w1list, w2list)

	Z = np.stack((w[0]*np.ones(W1.shape),W1,W2),axis=0)
	Z = Z.reshape((Z.shape[0],-1))
	Z = np.matmul(Xtest,Z) - Ytest.reshape((len(Ytest),1))
	Z = np.square(Z)
	Z = np.sum(Z, axis=0, keepdims=False)/Xtest.shape[0]
	Z = Z.reshape(W1.shape)
	
	if norm == 'L2':
		W_norm = np.square(W1) + np.square(W2)
	elif norm == 'L1':
		W_norm = np.abs(W1) + np.abs(W2)
	else:
		raise RuntimeError('Unimplemented norm. Please enter "l1" or "l2".')
		
	mse_ori = MSE(Xtest,Ytest,w)
	levels = [mse_ori, mse_ori+10]
	contour = ax.contour(W1, W2, Z, levels, colors='k')
	ax.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)

	if norm == 'L2':
		levels = [np.sum(np.square(w[1:]))]
	elif norm == 'L1':
		levels = [np.sum(abs(w[1:]))]
	else:
		raise RuntimeError('Unimplemented norm. Please enter "l1" or "l2".')
		
	contour = ax.contour(W1, W2, W_norm, levels, colors='r')
	ax.clabel(contour, colors = 'r', fmt = '%2.1f', fontsize=12)
	ax.plot(w[1],w[2],marker = ".",markersize=8)

	ax.set_title(norm + ' Result')
	ax.set_xlabel('$w_1$')
	ax.set_ylabel('$w_2$')
	ax.axis('square')
	return

def plotResult(W,Xtest,Ytest,
	levels=None,
	w1_range=(-4.0, 6.1, 100),
	w2_range=(-4.0, 6.1, 100),
	figtitle='dataset 1'):
	fig, (ax1, ax2) = plt.subplots(1, 2)
	display(W[0],Xtest,Ytest,ax1,norm='L1',w1_range=w1_range,w2_range=w2_range)
	display(W[1],Xtest,Ytest,ax2,norm='L2',w1_range=w1_range,w2_range=w2_range)
	plt.subplots_adjust(top=0.9)
	fig.suptitle(figtitle, fontsize=16, alpha=0.9,weight='bold')
	plt.show()

# def main():
# 	# how to load data
# 	filename = '../data/example_data.npz'
# 	dataset = np.load(filename)
# 	Xtrain,Ytrain,Xtest,Ytest = dataset['X_train'],dataset['y_train'],dataset['X_test'],dataset['y_test']

# 	# augment the features
# 	Xtrain = np.concatenate((np.ones((len(Xtrain),1)),Xtrain),axis=1)
# 	Xtest = np.concatenate((np.ones((len(Xtest),1)),Xtest),axis=1)
		
# 	# for l1 results
# 	W = [[ 2.91, 1.37, 2. ], [2.3,  1.39, 1.88]] 
# 	# for l2 results
# 	plotResult(W, Xtrain, Ytrain, figtitle='dataset 4')
# 	plt.show()
# 	return

# if __name__ == '__main__':
# 	main()