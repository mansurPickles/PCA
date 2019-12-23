import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def print_extra_values(X, pc1, pc2):
    #only use ths function for debugging and printing specific values to answer handout questions
    print(pc2.shape)
    print(X.shape)
    Y = np.dot(np.transpose(pc2), X)
    print(Y.shape)
    Y = np.transpose(Y)
    print(Y)

    eigen_sum = sum(eigen_value)
    temp = 0
    index = 1
    for i in eigen_value:
        temp += i / eigen_sum
        print(f' {index}, {round(temp, 2)}')
        index += 1

def svd(X):
    U, Sigma, V = np.linalg.svd(X)
    #sorting and finding the highest columns
    index = Sigma.argsort()[::-1][:2]
    #slicing V to the second column
    V = V[index]
    
    #calculate X_svd by dotting dataset with the V transpose
    X_svd = np.dot(X, V.T)
    return V.T

def pca(X):
    #create the covariance matrix
    covaraince = np.dot(X.T, X) / len(X)
    # print(f'printing covariance: {covaraince}')

    #store eigenvalue and eigenvector
    eigen_value, eigen_vector = np.linalg.eig(covaraince)

    print(f'printing eigenvalue: {eigen_value}')
    print(f'printing eigenvector: {eigen_vector}')

    #get the 2 highest eigenvalue vector columns
    index = eigen_value.argsort()[::-1][:2]

    #eigen_vector is now the best 2 columns
    eigen_vector = eigen_vector[:, index]

    #X_pca collect the 2 best eigen vectors
    X_pca = (eigen_vector[:, index])

    return X_pca


def mean(data):
    #calculate mean
    X = data
    means = []
    col_size = len(X[0])
    for i in range(0, col_size):
        means.append(np.mean(X[:, i]))

    print(f'printing means: {means}')
    return means

def standard_dev(data):
    #calcualte the standard deviation
    X = data
    col_size = len(X[0])
    stds = []

    for i in range(0, col_size):
        stds.append(np.std(X[:, i]))

    print(f'printing std deviation: {stds}')
    return stds

def norm (data):
    #normalize the data set
    col_size = len(data[0])

    stds = standard_dev(data)
    means = mean(data)

    for x in data:
        for item in range(col_size):
            x[item] = (x[item] - means[item]) / stds[item]

    print(f'printing normailized data: {data}')
    return data

#load data
datafile = 'cars.mat'
x = scipy.io.loadmat(datafile)
names = x['names']

#slice data and put it in varaible X
X = x['X'][:, 7:]

print(f'printing dataset:')
for x in X:
    print(x)

#normalize the dataset
X = norm(X)

#column size
col_size = len(X[0])

# create the covariance matrix
covaraince = np.dot(X.T, X) / len(X)
print(f'printing covariance: {covaraince}')

# store eigenvalue and eigenvector
eigen_value, eigen_vector = np.linalg.eig(covaraince)

#run PCA on X. store into pc1
pc1 = pca(X)

#run SVD on X. store into pc2
pc2 = svd(X)

projection1 = np.dot(X, pc1)
projection2 = np.dot(X, pc2)

print(f'projection of X onto PCA')
for i in projection1:
    print(i)
print('======'*40)
print(f'projection of X onto SVD')
for i in projection2:
    print(i)

#check orthogonal
print(f'checking if PCA pc are orthogonal: {round(np.dot(pc1[:,0], pc1[:,1]), 1)}')
print(f'checking if SVD pc are orthogonal: {round(np.dot(pc2[:,0], pc2[:,1]), 1)}')

#PCA is scattered in red
plt.scatter(projection1[:,0], projection1[:,1], color='red')

#SVD is scattered in blue
plt.scatter(projection2[:,0], projection2[:,1], color='blue')
plt.legend(('PCA', 'SVD'))


plt.ylim(-10,10)
plt.xlim(-10,10)
plt.title('SVD and PCA Using Normalized Data')

print(f'             PCA            ||            SVD                        ')
print('======'*10)

for i in range(len(pc1)):
    print(f'{round(pc1[i][0],10)}, {round(pc1[i][1], 10)}   ||   {round(pc2[i][0], 10)}, {round(pc2[i][1], 10)}')

plt.show()
