# Ganapathiraman, Bhavana
# 1001-748-622
# 2020_03_22
# Assignment-03-01

# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, transfer_function):
         """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
         self.weights.append(tf.Variable(np.random.randn(self.input_dimension,num_nodes)))
         self.biases.append(tf.Variable(np.random.randn(num_nodes,1)))
         self.activations.append(transfer_function.lower())
         self.input_dimension=num_nodes      

         

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
        """
        return self.weights[layer_number]
        
         

    def get_biases(self, layer_number):
         """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
         return self.biases[layer_number]
     

     

    def set_weights_without_biases(self, weights, layer_number):
         """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
         self.weights[layer_number]=weights
         

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases

        
  
        
    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        
        
    def sigmoid(self, x):
        print("inside sigmoid")
        return tf.nn.sigmoid(x)
        

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out   

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        X_duplicate=X
        for i in range(len(self.weights)):
            transpose_of_biases=tf.transpose(self.biases[i])
            y=(tf.matmul(X_duplicate,self.weights[i])+transpose_of_biases)
            
            if self.activations[i]=="sigmoid":
                a=self.sigmoid(y)
            
            elif self.activations[i]=="linear":
                a=self.linear(y)
                
            else:
                a=self.relu(y)
                
            X_duplicate=a
        return X_duplicate
    
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
         """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         
         """
         X_transpose=X_train.transpose()
         X_transpose_rows=np.size(X_transpose,0)

         for epoch in range(num_epochs):
            for index in range(0,X_transpose_rows,batch_size):
                 
                submatrix_X=X_train[index:(batch_size+index)]
                p=submatrix_X
                submatrix_Y=y_train[index:(batch_size+index)]
                submatrix_Y=submatrix_Y.transpose()
                with tf.GradientTape() as tape:
                    predictions = self.predict(p)
                    loss=self.calculate_loss(submatrix_Y,predictions)
                    w=self.weights
                    b=self.biases
                    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
    
                for layer in range(len(self.weights)):
                    self.weights[layer].assign_sub(alpha * dloss_dw[layer])
                    self.biases[layer].assign_sub(alpha * dloss_db[layer])

         

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        target=y
        number_of_samples=y.size
        number_of_errors=0
        out=self.predict(X)
        for i in range(0, number_of_samples):

            y_predicted=np.argmax(out[i,:])
            if(y_predicted != target[i]):
                number_of_errors+=1

        return (number_of_errors/number_of_samples)
    
    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        prediction=[]
        result=self.predict(X)
        for iter in result.numpy():  
            prediction.append(np.argmax(iter))
        confusion_matrix=tf.math.confusion_matrix(y,prediction)
        return confusion_matrix
        
        
