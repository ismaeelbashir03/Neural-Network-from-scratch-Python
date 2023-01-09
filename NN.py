import copy
import numpy as np
import pickle
# importing nnfs to test
import nnfs
# dataset for testing from nnfs
'''from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data'''

# testing initialisation
nnfs.init()

# defining our dense layer class
class DenseLayer():

    # constructor takes in the number of neurons and number of inputs on creation
    def __init__(self, num_inputs, num_nuerons, weight_reg_l1 = 0, bias_reg_l1 = 0, weight_reg_l2 = 0, bias_reg_l2 = 0):

        # weights are initially randomly set at standerd normal values in a matrix of 
        # size number of inputs (cols) by number of neurons (rows) 
        # (these are in transpose so we can do matrix multiplication later on). 
        # we want to scale this down to values that are less than one so we 
        # can multply by 0.01 
        self.weights = np.random.randn(num_inputs, num_nuerons) * 0.01

        # the bias is initially set to a zero matrix of size 
        # 1 (row) by number of neurons (cols), this is because 
        # there is one bias for each layer
        self.bias = np.zeros((1, num_nuerons))

        # here we define our lambda parameters for l1. this regularization
        # sums up all the weights and biases and adds it as a penalty to the
        # loss value, this stops few neurons doing most the work and becoming too big
        # (might lead to memorization). We also multiply our sum by lambda
        # to allow customization of how strict this penalty is.
        self.weight_reg_l1 = weight_reg_l1
        self.bias_reg_l1 = bias_reg_l1

        # here we set our lambda for l2. l2 regularization is very similar but we square
        # our weights and biases before we sum to make sure we give less penalty for 
        # smaller params. (this is mostly used than l1)
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l2 = bias_reg_l2

    # defining the function which passes nuerons forward through our NN
    def forward(self, inputs, training):
        
        # saving our inputs for optimisation
        self.inputs = inputs

        # defining our output as y = mx+c, where we multiply weights by inputs 
        # for each weight and input and add on a bias.
        self.output = np.dot(inputs, self.weights) + self.bias

    # creating the back propogation for the Dense Layer. This is done by going back through
    # our loss function and using the chain rule (we get the dirivitive of the current function 
    # and pass it backwards to be multiplied by the previous functions dirivitive.) 
    # we do all this to get the final dirivitve with respect to weights, 
    # biases and inputs. we can use this to minimise our loss function.
    def backward(self, prev_d):
        # here we get the diriviative with respects to the weights, which is
        # just the sum of the inputs multiplied by the previous dirivitves from the next layer.
        # the reason we just use the inputs is because the function to multiply the weights by 
        # input is differentiated to inputs when respect to weights (f(x) = xy, f'(x) = y).
        # Also we sum the inputs as we would have more than one dirivitive with respects to the 
        # weightfor the function if we did not.
        # we put the weights first in our dot product to match the shape of weights when we 
        # use them later
        self.d_weights = np.dot(self.inputs.T, prev_d)

        # to get the dirivitive of this function, it is just a sum, so the answer will be one
        # so we can just add up the previous dirivitives along the columns to get the dirivitive
        # for each bias.
        self.d_bias = np.sum(prev_d, axis = 0, keepdims = True)

        # we first check if we are using weight l1 regularization for this parameter.
        if self.weight_reg_l1 > 0:
            # if so we calculate the gradient for l1 as lambda*dy/dx(abs(param))
            # we can say this is lambda * (-1 or 1), (if param is positive or neg)
            d_l1 = np.ones_like(self.weights)
            # we check if its pos or neg and set -1 if needed
            d_l1[self.weights < 0] = -1 
            # we now add this to our weight derivitive as its needed to be taken into
            # account as we use the function at the end of the forward pass.
            self.d_weights += self.weight_reg_l1 * d_l1

        # we first check if we are using weight l2 regularization for this parameter.
        if self.weight_reg_l2 > 0:

            # we add this to our weight derivitive as its needed to be taken into
            # account as we use the function at the end of the forward pass.
            self.d_weights += 2*self.weight_reg_l2*self.weights

        # we first check if we are using bias l1 regularization for this parameter.
        if self.bias_reg_l1 > 0:
            # we use the same formula for the weights derivitive for l1
            d_l1 = np.ones_like(self.bias)
            d_l1[self.bias < 0] = -1 

            # we now add this to our bias divitive as its needed to be taken into
            # account as we use the function at the end of the forward pass.
            self.d_bias += self.bias_reg_l1 * d_l1
        
        # we first check if we are using bias l2 regularization for this parameter.
        if self.bias_reg_l2 > 0:

            # we add this to our bias derivitive as its needed to be taken into
            # account as we use the function at the end of the forward pass.
            self.d_bias += 2*self.bias_reg_l2*self.bias

        # to get the dirivitive of the inputs we need to sum all of the weights and then 
        # multiply them by the previous dirivitives. we can do this using dot product and 
        # the reason we just use the weights for inputs is the same as the reason for using 
        # the inputs for the weights, and we sum the weights for the same reason aswell.           
        self.d_inputs = np.dot(prev_d, self.weights.T)

    # we can return our parameters for analysing our network or for saving our model
    def get_params(self):
        return self.weights, self.bias

    # we should also be able to set parameters for loading in models
    def set_params(self, weights, bias):

        # we can just set these values
        self.weights = weights
        self.bias = bias

# we need to create a input layer for our model object as when we forward feed, our first
# layer doesnt have a previous layer to recive inputs from, so we can use this layer to do
# that. All we need to implement is that this layer is used to forward the inputs.
class InputLayer():

    def forward(self, inputs, training):
        self.output = inputs

# defining a class for a Rectified Linear Unit Activation function to be used 
# in the hidden layers of our NN. We will have to make another activation function
# for the final layer as we do not want to deal with negatives just turning into 0
# when calculating loss (loses meaning, we cannot work with it).
class ReLUActivation():
    
    # our activation function
    def forward(self, inputs, training):
        #storing the input for later
        self.inputs = inputs

        # comparing each input with zero, if it is bigger then 
        # we append it to output, if it is smaller we append zero.
        self.output = np.maximum(0, inputs)
    
    # creating the back propogation for the relu function. This is done by going back through
    # our loss function and using the chain rule (we get the dirivitive of the current function 
    # and pass it backwards to be multiplied by the previous functions dirivitive.) 
    # we do all this to get the final dirivitve with respect to weights, 
    # biases and inputs. we can use this to minimise our loss function.
    def backward(self, prev_d):
        
        # we create a copy of our previous dirivitive so we dont modify the original
        self.d_inputs = prev_d.copy()

        # we go through each of the inputs and check if they are smaller 
        # than zero, if so we make the value zero. we do this because when we want to 
        # differentiate a max function with zero, we either get zero or the constant
        #  multiplied by x. (f(x) = max(0,x), f'(x) = 1 if x > 0, or 0 otherwise, this is because
        # f(x) = x, f'(x) = 1). so if we did this manually, we would have a matrix of zeros and ones
        # we would then use to multiply our previous dirivitive, which will end up making values 
        # zero (if multiplied by zero), or make them the same (if multiplied by 1)
        self.d_inputs[self.inputs <= 0] = 0

    # we use a prediction function to get values we can calculate accuracy with
    # normally you would not use Relu for a final activation function but we will
    # get the predictions if someone wishes to do so in the future
    def predictions(self, outputs):
        # we can just return the outputs
        return outputs

#------ slow, use class ActivationSoftMaxLossCrossEntropy ------#
# our activation function for multiple choice classification. The softmax function 
# fixes all our issues with the previous function as it keeps the meaning of 
# the negative values while still getting rid of them, we do this by using
# exponential on all the inputs then normalising them by dividing them by the sum 
# of all the exponential values. If we did this as described above we may run into 
# overflow as exponential values can be huge so we can subtract the biggest value of our 
# inputs to the inputs to make them all atmost zero or negative, this way the only values we 
# can get for exponential values are between zero and one.
class SoftmaxActivation():

    # the activation function
    def forward(self, inputs, training):
        
        # saving the inputs for later
        self.inputs = inputs

        # here we use numpy to get the max, but since we are using a 2d array (batches of inputs),
        # we use axis 1 to get the max only within each row, and we keep the shape to subtract
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))

        # here we are doing the same with sum, axis 1 and we are keeping the dimensions
        norm_values = exp_values / np.sum(exp_values, axis = 1, keepdims= True)
        
        # setting the final answer as our output
        self.output = norm_values
    
    #------ slow, use class ActivationSoftMaxLossCrossEntropy ------#
    # creating the back propogation for the softmax function. This is done by going back through
    # our loss function and using the chain rule (we get the dirivitive of the current function 
    # and pass it backwards to be multiplied by the previous functions dirivitive.) 
    # we do all this to get the final dirivitve with respect to weights, 
    # biases and inputs. we can use this to minimise our loss function.
    def backward(self, prev_d):
        
        # first we create an array of shape of the previous dirivitive
        self.d_inputs = np.empty_like(prev_d)

        # here we loop through each output of the softmax function and the 
        # corrosponding previous dirivitive.
        for i, (each_output, each_prev_d) in enumerate(zip(self.output, prev_d)):
            # we first flatten each output so we can work with it later
            each_output = each_output.reshape(-1,1)

            # here we actually calculate the dirivitive. the dirivitive of this function 
            # ends up simplifying into ( output * kronecker delta ) - ( output * output )
            # A kronecker delta is 1 when i=j, but 0 when i != j. we get this as the 
            # dirivitive of a fraction is the dirivitive of the top mult by the bottom
            # subtracted by the dirivitive of the bottom mult by the top, we then divide this 
            # all by the bottom squared. this simplifies to 
            # ( dy/dx(e^input)*sum(e^input) - e^(input)*dy/dx(sum(e^input))/ sum(e^input)^2)
            # on the left of the subtraction the dirivitive is 1 when i=j for the output, but 
            # zero when i != j. this either gives (- e^(input)*dy/dx(sum(e^input))/ sum(e^input)^2)
            # which gives ( -e^input/sum(e^input) * e^input/sum(e^input) ), right hand side dirivitive
            # does nothing as it is just e. these bith are just softmax functions so we can say this 
            # simolifies to: -softmax*softmax, which is: softmax*(0-softmax)
            # the other case 1*e^output/sum(e^output) * (1 - e^output/sum(e^output)), which ends up being
            # softmax*(1-softmax). now we can put these together under the kronecker delta to be:
            # softmax(kronecker - softmax), which is: softmax*kronecker - softmax*softmax.
            # we can visualise kronecker delta. the kronecker delta can be visualised 
            # like a diagonal matrix we used before using np.eye
            """
            [
                [1,0,0],    1==1 so 1, 1!=2 so 0, 1!=3, so 0
                [0,1,0],    2!=1 so 0, 2==2 so 1, 2!=3, so 0
                [0,0,1],    3!=1 so 0, 3!=2 so 0, 3==3, so 1
            ]
            """
            # if we multiply this by our softmax output we get the left part of the answer, 
            # but we can do this all quickly using np.diagflat which gets a diagonal matrix of
            # size output and puts in the values into the diagonals, which is what we want.
            # for the right part of the answer we just need to dot product the matrices, 
            # but we need to tranform one of them to do the dot product.
            jacobian_matrix = np.diagflat(each_output) - np.dot(each_output, each_output.T)

            # now that we have the dirivitive of this function, we can multiply it by each 
            # gradient passed in to get a value to send back.
            self.d_inputs[i] = np.dot(jacobian_matrix, each_prev_d)
    
    # we use a prediction function to get values we can calculate accuracy with
    def predictions(self, outputs):
        
        # we use argmax to get the index of the one hot encoded outputs and 
        # put them in a list to be returned
        return np.argmax(outputs, axis = 1)

# this activation function is used to calculate an output for binary regression
# the formula gives us values between one and zero, which we need as our neurons
# will only give values of 1 or zero depending if we have identified something or 
# not.
class SigmoidActivation():

    # here we calculate the sigmoid output
    def forward(self, inputs, training):
        # we save the inputs for later
        self.inputs = inputs

        # getting the output of the sigmoid function with the input
        self.output = 1/(1 + np.exp(-inputs))

    # the dirivitive of the sigmoid function is very common, it ends up being
    # sigmoid * (1-sigmoid). we get this by taking th denominator up to get negative 
    # powers and solving to get: (1+e^-input)*(e^output * dy/dx(-output)) we can simplify
    # a bit to get: e^output/(1+e^outout)^2 = sigmoid * (1 - sigmoid), our output 
    # is the sigmoid here.
    def backward(self, prev_d):
        # using the formula
        self.d_inputs = prev_d * (1-self.output)*self.output

    # we use a prediction function to get values we can calculate accuracy with
    def predictions(self, outputs):
        # if our outputs are above 0.5 they are true, false if vice versa, we then 
        # multiply by 1 to get in numerical values
        return (outputs > 0.5) * 1

# this activation class is used for scalar data (predicted specific values)
# this is easy to code as the function is just y = x, so the input is unchanged
class LinearActivation():

    def forward(self, inputs, training):

        # here we store the inputs for later
        self.inputs = inputs

        # we just set the output as the input
        self.output = self.inputs
    
    # for the dirivitive of y = x, it is just 1 so we do 
    # nothing to the previous dirivitive
    def backward(self, prev_d):

        # 1 * previous dirivitive
        self.d_inputs = prev_d.copy()

    # we use a prediction function to get values we can calculate accuracy with
    def predictions(self, outputs):
        # for this function we can just return the output as we want the scalar value
        return outputs


# creating a general class for accuracy
class Accuracy():
    
    # calculating accuracy
    def calculate(self, predictions, real_y):

        # first we get the comparison results (if correct true, if wrong false)
        comparisons = self.compare(predictions, real_y)
        
        # after we calculate the actual accuracy by getting the mean comparisons
        accuracy = np.mean(comparisons)

        # saving values for the bacth mean accuracy
        self.accumulated_sum += np.sum(comparisons)

        self.accumulated_count += len(comparisons)    

        # we then just return our accuracy
        return accuracy
    
    # we run this every epoch to the mean accuracy
    def calculate_accumulated(self):

        # calculating the mean bach accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    
    # reseting our accumulated batch accuracy, ran every epoch
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# this accuracy is used for binary regression (each neuron has a classifier)
# and classifers all neurons are use for one classifier
class CategoricalAccuracy(Accuracy):

    def __init__(self, *,binary = False):

        self.binary = binary

    # we make this class so we can sub accuracy models into our
    # model without having to check if they have an init ethod or not
    def init(self, real_y):
        pass

    # here we get our truth or false list
    def compare(self, predictions, real_y):

        # if our real y is in 2d from, we change it to be the
        # index of the correct category in list form
        # we do this only for multiple choice classifiers
        if not self.binary and len(real_y.shape) == 2:
            real_y = np.argmax(real_y, axis = 1)

        # we then return the true or false list if the 
        # predictions equal the real y
        return (predictions == real_y)

# here we will make an accuracy class fro regression
class RegressionAccuracy(Accuracy):

    def __init__(self):
        self.precision = None

    # this method will initialise our precision value
    def init(self, real_y, precision_strength = 250, re_init = False):

        if self.precision is None or re_init:
            self.precision = np.std(real_y) / precision_strength

    # here we compare our predictions to the real y values
    def compare(self, predictions, real_y):
        # if the difference of our predictions and the real y values is 
        # less that the precision/limit we calculated earlier, we set it to true
        return np.absolute(predictions-real_y) < self.precision


# a generic loss class others can inherit from
class Loss():

    # we use this function to set trainable layers for our loss functions to use
    def remember_trainable_layers(self, trainable_layers):
        # we simply save this list
        self.trainable_layers = trainable_layers
    
    # calculating the loss from the 
    def calculate(self, output, y, include_regularization = False):

        # getting our losses in a vector format (batches of inputs)
        sample_loss = self.forward(output, y)

        # getting the mean of this vector to get one loss value
        batch_loss = np.mean(sample_loss)

        # adding up our batch losses and count for mean batch loss
        self.accumulated_sum += np.sum(sample_loss)
        self.accumulated_count += len(sample_loss)

        # setting our self loss to be used to print later
        self.loss = batch_loss

        # we check if we are not using regularization
        if not include_regularization:
            return batch_loss

        # we return our batch loss and also out regularization loss
        return batch_loss, self.regularization_loss()

    # this runs every epoch
    def calculate_accumulated(self, *, include_regularization = False):
        
        # we get the mean loss of the batches
        data_loss = self.accumulated_sum / self.accumulated_count


        if not include_regularization:
            # If just data loss - return it ​if not ​include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss for each epoch
    def new_pass(self): 
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):

        reg_loss = 0

        # we loop through each trainable layer
        for layer in self.trainable_layers:

            # first we check if we are using a weight strength of more than zero
            # (if its being used)
            if layer.weight_reg_l1 > 0:
                # here we update our regularization loss with the sum of the weights
                # multiplied by our strength value
                reg_loss += np.sum(np.abs(layer.weights)) * layer.weight_reg_l1

            # first we check if we are using a weight strength of more than zero
            # (if its being used)
            if layer.weight_reg_l2 > 0:
                # here we update our regularization loss with the sum of the weights squared
                # multiplied by our strength value
                reg_loss += np.sum(layer.weights*layer.weights) * layer.weight_reg_l2

            # first we check if we are using a bias strength of more than zero
            # (if its being used)
            if layer.bias_reg_l1 > 0:
                # here we update our regularization loss with the sum of the bias
                # multiplied by our strength value
                reg_loss += np.sum(np.abs(layer.bias)) * layer.bias_reg_l1

            # first we check if we are using a bias strength of more than zero
            # (if its being used)
            if layer.bias_reg_l2 > 0:
                # here we update our regularization loss with the sum of the bias squared
                # multiplied by our strength value
                reg_loss += np.sum(layer.bias*layer.bias) * layer.bias_reg_l2
        
        # we return our caclulated regularization loss
        return reg_loss

#------ slow, use class ActivationSoftMaxLossCrossEntropy ------#
# categorical cross entropy loss, 
# (negative natural log of the value predicted for the correct index) when 
# using one hot encoding ( [0,0,0,1] ). this is useful as when our predicted 
# value is closer to one the log value is closer to zero and when the predicted 
# value approaches zero the log value approaches negative infinity 
# (we get rid of negative by multiplying by -1).
class CategoricalCrossEntropyLoss(Loss):

    # here we calculate the loss
    def forward(self, pred_y, real_y):
        # getting the number of the batches
        num_batches = len(pred_y)

        # clipping our values to be between -1e-7 and 1e-7 because we dont want a 
        # value that gives zero as our log value will then be neg infinity, 
        # (which we cannot use), so -1-e7 is small enough to set a minimum.
        clipped_pred = np.clip(pred_y, 1e-7, 1-1e-7)

        # we next need to check if we are passed a scalar value or a 2d array 
        # as real y values ([0,1,0] or [1,0],[0,1],[1,0])

        # if we are using scalar values:
        if len(real_y.shape) == 1:
            # we get the confidence of the index which was supposed to be 1 (true) 
            # we do this by using np indexes to get all the rows with 
            # range(first argument) ( [0,1,2,3...] ) (gets all the rows)
            # then we get the columns of the correct index from the real y ([0,1,0])
            real_confidence = clipped_pred[range(num_batches), real_y]

        # if we are using a 2d array input for the real y
        elif len(real_y.shape) == 2:
            # we get the confidence of the correct index by multiplying the 
            # clipped predicted values by the real y values, we then sum the columns,
            # axis 1 (cols). doing all this multiplies each value in the predicted by the 
            # corrosponding value in the real y, which will give all zeros apart from
            # the values we need as they will be multiplied by 1 (they are correct). then 
            # by summing the values in the cols, we get a 1d array with only our confidence 
            # for our real index.
            real_confidence = np.sum(clipped_pred*real_y, axis = 1)

        # getting the negative log of each value in our confidences
        neg_log_loss = -np.log(real_confidence)

        # saving the loss to be used to print
        self.loss = neg_log_loss

        # returning the loss
        return neg_log_loss

    #------ slow, use class ActivationSoftMaxLossCrossEntropy ------#
    # creating the back propogation for the loss function. This is done by going back through
    # our loss function and using the chain rule (we get the dirivitive of the current function 
    # and pass it backwards to be multiplied by the previous functions dirivitive.) 
    # we do all this to get the final dirivitve with respect to weights, 
    # biases and inputs. we can use this to minimise our loss function.
    def backward(self, output, real_y):
        
        # we need to get the number of batches for later, we get this as 
        # the output is a 2d array
        num_samples = len(output)

        # we also need the number of labels in each of the derivitives
        num_labels = len(output[0])

        # checking if the real y values are in 2d array form to be used 
        # with the output of our model
        if len(real_y.shape) == 1:
            # np.eye creates a diagonal with columns of size num labels
            # then we pick the index of the row of the real y index to get a 
            # list of zeros with a one at the index real_y
            """
            e.g.
            [
                [1,0,0],
                [0,1,0],
                [0,0,1]
            ]
            if real_y = 2 we get [0,1,0]
            """ 
            real_y = np.eye(num_labels)[real_y]

        # here we get the dirivitive by using the simplified version of the dirivitive of 
        # ( -real_y * ln(output) ), this simifies to output / real_y
        self.d_inputs = - (real_y / output)

        # we then normalise our gradient by dividing by the number of batches
        # we do this as we have to sum our gradients later in optimization, so this will
        # make the numbers easier to work with
        self.d_inputs = self.d_inputs / num_samples

# this loss is similar to categorical cross entropy but this time we do the negative log
# on the other classes too, since there is only one other class (0). we can say the forumula is
# (real_y)*-log(pred_y) + (1-real_y)*(-log(1-pred_y)), we use 1-real/pred_y becuase they are the 
# inverse of the other class and there are only two of them.
class BinaryCrossEntropyLoss(Loss):

    # here we use the formula
    def forward(self, pred_y, real_y):
        # we need to clip the values so we dont have log(0), which gives
        # negative infintity.
        clipped_pred_y = np.clip(pred_y, 1e-7, 1-1e-7)

        # here we calculate the loss useing the formula in batch form
        # (the formula has been simplified to make it easier to code)
        batch_loss = -(real_y*np.log(clipped_pred_y) + (1-real_y)*np.log(1-clipped_pred_y))

        # since we have multiple neurons for each classification, we need one loss
        # value for each neuron so get the mean, axis -1 is used to get the last dimension
        batch_loss = np.mean(batch_loss, axis = -1)

        # we return this value to be printed
        return batch_loss

    def backward(self, outputs, real_y):

        # first we get the number of batches
        num_batches = len(outputs)

        # we then need the number of outputs in every batch
        num_outputs = len(outputs[0])

        # we need to clip our output of our activation function so we 
        # dont divide by zero accidently
        clipped_outputs = np.clip(outputs, 1e-7, 1-1e-7)

        # here we calculate the dirivitive. we gte this by first diriving each neuron first
        # this gives:
        # -real_y * dy/dx(log(pred_y)) - (1-real_y)*dy/dx(log(1-pred_y)) which gives:
        # -real_y/pred_y - (1-real_y)/(1-pred_y)*(0-1) = -(real_y/pred_y - (1-real_y)/(1-pred_y))
        # now that we have the equation for each neuron we can get the dirivitive of the sum
        # which is: 1/num_neurons * sum(each neuron), this ends up being 1/num_neurons as 
        # dy/dx(each nuron) = 1. we multiply these two to get:
        # 1/num_neurons * -(real_y/pred_y - (1-real_y)/(1-pred_y))
        self.d_inputs = -(real_y/clipped_outputs - (1-real_y) / (1-clipped_outputs)) / num_outputs

        # here we normalise the dirivitive so the sum is small when we optimise
        self.d_inputs = self.d_inputs / num_batches

# for scalar value predictions we cant use cross entropy so we can use mean squared
# error, this a common loss function which gets the difference of the real_y and the
# predicted y and squares the result, then this is summed for each difference and 
# averaged. This function penalises you more the more further away you are from the 
# target: 1/num_loss(sum(real_y-pred_y)**2)
class MeanSquaredErrorLoss(Loss):
    # here we implement the function
    def forward(self, pred_y, real_y):

        # we use axis -1 here becuse we could have any number of batches
        # in numpy -1 is a variable value
        self.batch_loss = np.mean((real_y-pred_y)**2, axis = -1)

        return self.batch_loss

    # to calculate the dirivitive of the mean squared error we can take out 
    # the division and get rid of the sum, this gives:
    # 1/num_loss(dy/dx((real_y-pred_y)**2)
    # = 1/num_loss(2(real_y-pred_y)*dy/dx(real_y-pred_y))
    # = 1/num_loss(2(real_y-pred_y)*(0-1))
    # = -1/num_loss(2(real_y-pred_y)
    def backward(self, pred, real_y):
        
        # first we get the number of batches
        num_batches = len(pred)

        # we then get the number of inputs in each batch
        # (num of loss)
        num_outputs = len(pred[0])

        # here we use our formula
        self.d_inputs = -2 * (real_y - pred) / num_outputs

        # we need to normalise the gradient to make optimisation easier
        self.d_inputs = self.d_inputs / num_batches

# this is another scalar loss function, but this is used less than MSE
# this function penalises error linearly and produces sparser results
# while also being robust to outliers, this can be good or bad depending
# on the case. This formula is almost the same as MSE but instead of squaring 
# the difference we just get the absolute value
class MeanAbsoluteErrorLoss(Loss):

    # here we use the formula
    def forward(self, pred_y, real_y):
        
        batch_loss = np.mean(np.abs(real_y-pred_y))

        return batch_loss
    
    def backward(self, pred, real_y):

        # here we gte the number of batches and the number
        # of output in each batch
        num_batches = len(pred)
        num_outputs = len(pred[0])

        # here we use our formula
        self.d_inputs = np.sign(real_y-pred) / num_outputs

        # we need to normalise for our optimisation
        self.d_inputs = self.d_inputs / num_batches


# since in our backward functions for cross entropy and softmax we used
# unefficient ways of calculating the gradient. Here we will combine the calculation
# as it will simplify down nicely to be computed faster.
class ActivationSoftMaxLossCrossEntropy():

    # we first define our loss and final activation function
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = CategoricalCrossEntropyLoss()
    
    def backward(self, predicted, real_y):
        
        # first we get the number fo batches we are working with
        num_batches = len(predicted)

        # here we check if the real y values are in one hot encoded format
        if len(real_y.shape) == 2:
            # here we change them to just discrete values (e.g [1,3,2,0]), we use 
            # axis 1 as this is in 2d array format so we apply the function to each column
            real_y = np.argmax(real_y, axis = 1)

        # we take a copy of the predicted values so they can remain unchanged
        self.d_inputs = predicted.copy()

        # here we get the values of the correct index and subtract them from 
        # the real value they should be (this always ends up being 1)
        # we do this as this is the simplified version of the dirivitive of 
        # the cross entropy loss and softmax activation functions. 
        # the equation is dirivitive of loss with respect to predicted values
        # mult by the dirivitive of softmax with respect to output of softmax (chain rule)
        # we can then say the answer of softmax is the predicted value so we can sub that in.
        # we also know that dirivitive of loss is just sum of real y over predicted y so we can also
        # sub this in. We have to take into account both cases for our krocecker delta values so we can
        # multiply both cases to the loss dirvitive and subtract them (left side with sum i=j and right 
        # side with sum i!=j). this gives: 
        # - real_y/pred_y * kronecker(i=j) - sum(real_y/pred_y) * kronecker(i!=j). since on 
        # the left hand side we are only dealing with i=j the sum is useless as its only one total value
        # so we can get rid of it. we can now cancel both fractions as they are both multiplying
        # by the predicted value and we can make the subtraction an addition with the negative
        # to get: real_y*(1-pred_y) + sum(real_y)*pred_y, we can multiply out and we get: 
        # -real_y+real_y*pred_y + sum(real_y)*pred_y, since the left is i=j and right is sum of i!=j
        # we can join these to get a full sum and get: -real_y + sum(real_y)*pred_y, where the right is
        # the sum for all i and j. and since we are summing one hot encoeded vectors multiplied we will only get
        # the value 1 for the real index and the rest will be zero, so we can simplify the sum to 1, 
        # as 1+0+0... = 1. so we get -real_y + pred_y, which gives pred_y - real_y. this formual is much simplier
        # than the loop used in the previous plus all the other calculations in both functions and even tho it was 
        # alot of work to get here it was worth it fot the efficiency.
        self.d_inputs[range(num_batches), real_y] -= 1

        # we can now normalise our gradients so when we optimise we can 
        # deal with smaller numbers when we sum
        self.d_inputs = self.d_inputs/num_batches

# optimizer class for Stochastic Gradient Descent, here we take in a learning rate
# and this is multiplied by the gradient of the weights and biases and is
# subtracted to change the values to lower the loss. We do this as the gradient shows the 
# direction of the function so we want to lower the function of loss so we give it negative
# gradient with a multiple of learning rate
class SGDOptimizer():

    # constructor to initialise variables passed in
    def __init__(self, learning_rate=1.0, decay_rate=0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.mom = momentum

    # this is run at the beginning to setup the decay
    def pre_update_params(self):
        # we check if a decay was used
        if self.decay_rate:
            # we multiply the learning rate by 1 / (1 + decay_rate mult by the iteration), 
            # this makes suure we always decrease the learning rate (the plus 1 at the bottom)
            # and we decay less as time goes on as we multiply by the iteration, the higher the 
            # iteration, the lower the total decay
            self.current_learning_rate = self.learning_rate * (1/(1 + self.decay_rate*self.iterations))


    # here we adjust the weights and biases with the learning rate and gradient
    def update_params(self, layer):
        
        # we check if we are using momentum
        if self.mom:
            
            # we check if the layer class has the weight momentmums variable, and if not we create one
            if not hasattr(layer, "weight_moms"):
                # we create a weight and bias momentum variable for the class
                layer.weight_moms = np.zeros_like(layer.weights)
                layer.bias_moms = np.zeros_like(layer.bias)

            # here we use the momentum formula, momentum multiplied by previous weight update to be 
            # subtracted by the learning rate multiplied by the weight gradients. the momentum helps us 
            # to stop being stuck in local minimums as it remembers the direction of the previous weight change
            # this can help push our weight out of the 'upper hill' of a local minimum to get to the global minimum
            weight_updates = self.mom*layer.weight_moms - self.current_learning_rate*layer.d_weights
            # updating the weight momentum in the layer to be used next time
            layer.weight_moms = weight_updates
            

            # here we do the same but for bias
            bias_updates = self.mom*layer.bias_moms - self.current_learning_rate*layer.d_bias
            # updating the weight momentum in the layer to be used next time
            layer.bias_moms = bias_updates

        # if we are not using momentum we can use the original learning rate update
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_bias
        
        # we update our weights and bias in the layer
        layer.weights += weight_updates
        layer.bias += bias_updates

    # this is ran after we update the params of the layers
    def post_update_params(self):
        # we update our iteration counter for the next decay
        self.iterations += 1

# optimizer class for the adaptive Gradient Descent, here we have a cache value
# which stores the previous gradients added to a parameter squared. this is then used to 
# make the function (learning rate * gradient) / sqrt(cache)+epsilon. This makes it so that
# parameters have a soft limit on how much they can change, if you change them alot on one
# step the next step the change is divided to be smaller. this allows other parameters to 
# catch up to the big changing weights and allows more neurons to be used. But this can halt
# learning as eventually there will be little to no change in training, so this is used for
# specific cases normally. We use a epsilon variable in the function to stop division by zero
# eps will be set to 1e-7 to do this.
class ADAGradOptimizer():

    # constructor to initialise variables passed in
    def __init__(self, learning_rate=1.0, decay_rate=0.0,epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.eps = epsilon

    # this is run at the beginning to setup the decay
    def pre_update_params(self):
        # we check if a decay was used
        if self.decay_rate:
            # we multiply the learning rate by 1 / (1 + decay_rate mult by the iteration), 
            # this makes suure we always decrease the learning rate (the plus 1 at the bottom)
            # and we decay less as time goes on as we multiply by the iteration, the higher the 
            # iteration, the lower the total decay
            self.current_learning_rate = self.learning_rate * (1/(1 + self.decay_rate*self.iterations))


    # here we adjust the weights and biases with the learning rate and gradient
    def update_params(self, layer):
        
            
        # we check if the layer class has the weight cache variable, and if not we create one
        if not hasattr(layer, "weight_cache"):
            # we create a weight and bias cache variable for the class
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # here we update the cache to store the current change with bias and weights
        layer.weight_cache += layer.d_weights**2
        layer.bias_cache += layer.d_bias**2
        
        # here we update the weights and biases with our cache limiting the change
        layer.weights += -self.current_learning_rate*layer.d_weights / (np.sqrt(layer.weight_cache)+self.eps)
        layer.bias += -self.current_learning_rate*layer.d_bias / (np.sqrt(layer.bias_cache)+self.eps)

    # this is ran after we update the params of the layers
    def post_update_params(self):
        # we update our iteration counter for the next decay
        self.iterations += 1

# this optimizer is similar to the ada grad optimizer except we calculate the 
# cache differently, instead we use the formula: 
# cache = rho*cache + (1-rho)*gradient^2. rho is used to set the ratio between
# cache and gradient. this method creates a smoother jump of learning rates as
# we use the previous cache as a percentage replacement of our next gradient change
# since we carry alot of the gradient into the next cache we have to lower the learning rate
# as this is too big, a common value used as default is 0.001
class RMSPropOptimizer():

    # constructor to initialise variables passed in
    def __init__(self, learning_rate=0.001, decay_rate=0.0,epsilon = 1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.eps = epsilon
        self.rho = rho

    # this is run at the beginning to setup the decay
    def pre_update_params(self):
        # we check if a decay was used
        if self.decay_rate:
            # we multiply the learning rate by 1 / (1 + decay_rate mult by the iteration), 
            # this makes suure we always decrease the learning rate (the plus 1 at the bottom)
            # and we decay less as time goes on as we multiply by the iteration, the higher the 
            # iteration, the lower the total decay
            self.current_learning_rate = self.learning_rate * (1/(1 + self.decay_rate*self.iterations))


    # here we adjust the weights and biases with the learning rate and gradient
    def update_params(self, layer):
        
        # we check if the layer class has the weight cache variable, and if not we create one
        if not hasattr(layer, "weight_cache"):
            # we create a weight and bias cache variable for the class
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # here we set the cache to store a percent of the gradient and previous cache for 
        # the weights and biases
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*layer.d_weights**2
        layer.bias_cache = self.rho*layer.bias_cache + (1-self.rho)*layer.d_bias**2
        
        # here we update the weights and biases with our cache limiting the change
        layer.weights += -self.current_learning_rate*layer.d_weights / (np.sqrt(layer.weight_cache)+self.eps)
        layer.bias += -self.current_learning_rate*layer.d_bias / (np.sqrt(layer.bias_cache)+self.eps)

    # this is ran after we update the params of the layers
    def post_update_params(self):
        # we update our iteration counter for the next decay
        self.iterations += 1

# this optimizer takes the RMSProp optimizer cache and adds the momentum from SGD.
# we first use beta_1 as our ratio of previous momentum to current gradient. we do this by 
# using a similar formula to RMS prop, but without squaring: beta_1*momentum + (1-beta-1)*gradient
# then we correct our momentums by dividing them by 1-beta_1^(steps+1), beta_1^(steps+1) 
# approaches 0 as the steps increase this results in a division of a number smaller than 
# one to increase the momentum at the beginning but as the steps increase we tend to one 
# which ends up dividing by one, which does little change to the momentum in the later steps.
# the reason we do this is to jump start the momentum from the beginning to make up for starting 
# it with zeros. this can increase the speed of training by a huge margin. we then use our cache
# formula from RMS prop (with squares) and using beta_2 this time we then also jump start these values 
# by using the same formula also with beta_2. finally we use all this to calculate the weights and 
# biases using a formula similar to RMS but we multiply learning rate by momentum instead of gradient:
# -learning_rate*jumpstart_momentum / (sqrt(jumpstart_cache)+epsilon), this will give us a change that uses 
# momentum but also soft locks the change to let other weights keep up and make more neurons useful, we also
# see improved speeds thanks to our jump starts of momentum and cache.
class AdamOptimizer():

    # constructor to initialise variables passed in
    def __init__(self, learning_rate=0.001, decay_rate=0.0,epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.eps = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # this is run at the beginning to setup the decay
    def pre_update_params(self):
        # we check if a decay was used
        if self.decay_rate:
            # we multiply the learning rate by 1 / (1 + decay_rate mult by the iteration), 
            # this makes sure we always decrease the learning rate (the plus 1 at the bottom)
            # and we decay less as time goes on as we multiply by the iteration, the higher the 
            # iteration, the lower the total decay
            self.current_learning_rate = self.learning_rate * (1/(1 + self.decay_rate*self.iterations))


    # here we adjust the weights and biases with the learning rate and gradient
    def update_params(self, layer):
        
            
        # we check if the layer class has the weight cache variable, and if not we create one
        if not hasattr(layer, "weight_cache"):
            # we create a weight and bias cache variable for the class
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_moms = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
            layer.bias_moms = np.zeros_like(layer.bias)

        # here we set the momentum weights and biases with the current gradient
        # (similar as RMS prop optimizer cache)
        layer.weight_moms = self.beta_1*layer.weight_moms + (1-self.beta_1)*layer.d_weights
        layer.bias_moms = self.beta_1*layer.bias_moms + (1-self.beta_1)*layer.d_bias

        # here we jump start our momentums (these will be almost unchanged as steps increase)
        weight_moms_corrected = layer.weight_moms / (1 - self.beta_1 ** (self.iterations+1))
        bias_moms_corrected = layer.bias_moms / (1 - self.beta_1 ** (self.iterations+1))

        # we caclulate our cache for weights and biases
        layer.weight_cache = self.beta_2*layer.weight_cache + (1-self.beta_2)*layer.d_weights**2
        layer.bias_cache = self.beta_2*layer.bias_cache + (1-self.beta_2)*layer.d_bias**2

        # here we jump start our cache (these will be almost unchanged as steps increase)
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations+1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations+1))

        # here we update our weights and biases using our corrected momentum and cache
        layer.weights += -self.current_learning_rate*weight_moms_corrected / (np.sqrt(weight_cache_corrected)+self.eps)
        layer.bias += -self.current_learning_rate*bias_moms_corrected / (np.sqrt(bias_cache_corrected)+self.eps)



    # this is ran after we update the params of the layers
    def post_update_params(self):
        # we update our iteration counter for the next decay
        self.iterations += 1

# this another version of regularization we can use. this, with a given rate, will
# 'turn off' neurons during training with the given to allow our network to have more
# nuerons learn the same thing to allow a deeper understanding of our problems function
# instead of just one neuron memorising what to do (over-fitting). This also helps with 
# neurons just relying on previous neurons output and requires them to learn the function 
# better. To do this we will create a bernoulli from a binomial function from numpy which
# will give us an output of 1's and 0's with the rate we pass into it, this can be multiplied
# by our layers output to 'turn off' some neurons. Since we are zeroing the output of neurons 
# to turn them off, this creates a problem in testing as we run into the issue of the network 
# recieving large outputs, when the outputs are summed, sent to the next layer after being 
# trained with normally lower values as the training outputs had zeros in the sum that made them 
# smaller. To compinsate for this difference we can divide our dropout array, with the inverse 
# rate of dropout. this will scale up the values that have not been zeroed, giving bigger sums in 
# training to match the testing.
class DropoutLayer():

    # constructor to get rate
    def __init__(self, rate):
        # here we invert the rate for later use
        self.rate = 1-rate

    # here we actually 'turn off' some neurons
    def forward(self, inputs, training):
        # we save the input for later
        self.inputs = inputs    

        # we need to check if we are training or not and if so
        # we just set our out to our input and return
        if not training:
            self.output = inputs.copy()
            return 

        # here we create our dropout array to be multiplied by our input, 
        # we pass in our number of chances (number of times done in each trial), 
        # probability, and number of trials.
        self.binary_mask = np.random.binomial(1, self.rate, size = self.inputs.shape) / (self.rate)

        # here we multiply the output of the neurons by our dropout array
        self.output = inputs * self.binary_mask

    # our back propigation for this is just the dirivitive of 0, which is zero
    # or just 1 as the function if 1 is: input/1-q, which is (1/1-q)*dy/dx(input)
    # which gives = 1/1-q * 1 = 1/1-q, we can just multiply by our dropout array 
    # to get the same effect
    def backward(self, prev_d):
        self.d_inputs = prev_d * self.binary_mask

# here we create a model to class to build our network
class Model():

    def __init__(self):
        # we initialise a list of layer to create a frame for our network
        self.layers = []
        # initialising our optimised softmax and categorical cross entropy function
        self.softmax_cross_entropy_output = None

        # initialising loss, optimizer and accuracy
        self.loss = None
        self.optimizer = None
        self.accuracy = None

    # we use this function to add layers to our network
    def add(self, layer):

        # updating our list with the new layer passed in
        self.layers.append(layer)

    # we use this function to set our optmisation and loss function*
    def set(self, *, loss = None, optimizer = None, accuracy = None):

        # we are going to check if the loss is empty, if so we set it
        if self.loss == None:
            self.loss = loss

        # we are going to check if the optimizer is empty, if so we set it
        if self.optimizer == None:
            self.optimizer = optimizer

        # we are going to check if the accuracy is empty, if so we set it
        if self.accuracy == None:
            self.accuracy = accuracy

    # this function is used to train our model with data
    def train(self, X, real_y, *, epochs = 1, print_every=1, validation_data = None, batch_size = None):

        # we first initialise our accuracy percision
        self.accuracy.init(real_y)

        # default vaLue if batch size not set
        train_steps = 1
        
        # checking if we put in a batch size
        if batch_size is not None:
            
            # Calculating number of steps
            # we do this by dividing number of data by the batch size
            train_steps = len(X) // batch_size

            # Dividing rounds down. If there are some remaining 
            # # data, but not a full batch, this won't include it 
            # # Add 1 to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1
        
        # we loop throught the number of epochs, starting from 1 (for readability)
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects 
            self.loss.new_pass()
            self.accuracy.new_pass()

            # looping for each batch
            for step in range(train_steps):
                
                if step == 468:
                    print("HERE")

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = real_y

                else:
                    # Otherwise slice a batch ​else​:
                    batch_X = X[step * batch_size:(step+1) * batch_size] 
                    batch_y = real_y[step * batch_size:(step+1) * batch_size]


                # we run our forward function which will pass through our data through the
                # the network
                output = self.forward(batch_X, training = True)

                # here we calculate the data loss and regularization loss
                data_loss, reg_loss = self.loss.calculate(output, batch_y, include_regularization = True)

                # we add up both our losses to get a total loss
                loss = data_loss + reg_loss

                # here we get the predictions from our last activation function in a form
                # we can use to calculate accuracy
                predictions = self.output_layer_activation.predictions(output)

                # here we calculate the accuracy 
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # we next need to back propigate to get our gradients
                self.backward(output, batch_y)

                # finally we can optimize our trainable layers for the next epoch
                # we first run the setup for the optimization
                self.optimizer.pre_update_params()

                # we can now loop through all the trainable layers and apply 
                # this optimization
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                
                # finally we can update our iterations in our optimizations
                self.optimizer.post_update_params()

                # we can now print all our information
                # if the step / print_every has a remainder of 0, we print
                # e.g. 100%100 = 0, 200%100 = 0, or we are at the last step
                if not step % print_every or step == train_steps - 1:
                    print(
                        f"step {step} " +
                        f"acc: {accuracy:.3f} " +
                        f"loss: {loss:.3f} " +
                        f"data loss: {data_loss:.3f} " +
                        f"reg loss: {reg_loss:.3f} " +
                        f"learn rate: {self.optimizer.current_learning_rate}"
                    )

            # Getting epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization= True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss 
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            # printing epoch data
            print(f' training, ' +
                f' acc: {epoch_accuracy:.3f}, ' +
                f' loss: {epoch_loss:.3f} (' +
                f' data_loss: {epoch_data_loss:.3f}, ' +
                f' reg_loss: {epoch_regularization_loss:.3f}), ' + 
                f' lr: {self.optimizer.current_learning_rate}'
            )

            # here we use our validation data if present
            if validation_data != None:
                
                # we can evaluate our validation data.
                #  '*' unpacks the validation data for us
                self.evaluate(*validation_data, batch_size=batch_size)

    # we will use this function in the training function to set the next and previous
    # layer
    def finalize(self):

        # we first initialise the input layer used to start the forward feed
        self.input_layer = InputLayer()

        # we then get the number of actual layers we are using
        layer_count = len(self.layers)

        # we initialise our list of trainable layers
        self.trainable_layers = []

        # we go through each layer
        for i in range(layer_count):
            
            # if we are at the first layer
            if i == 0:
                # we set the previous layer to the input layer and next 
                # layer as i+1 (next layer in the loop)
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # if we are not at the last or the first layer 
            elif i < layer_count-1:

                # here we set the next layer to be i+1 and the previous
                # layer to be i-1
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            # if we are in the last layer
            else:
                # we set the previous layer as i-1 and the next layer
                # as the loss class
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                
                # we can save our final activation layer for later
                self.output_layer_activation = self.layers[i]

            # for back propigation we need to know which layers are trainable
            # (layers with weights and biases) we can do this with the hasattr 
            # function
            if hasattr(self.layers[i], "weights"):
                # if this layer has weights (meaning itll also have biases)
                # we add it to our trainable layer list
                self.trainable_layers.append(self.layers[i])

        # we check if we have a loss function (if we imported a model)
        if self.loss != None:
            # we update the the loss function with the list of trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

        # we need to check of we are classifying and uses categorical cross entropy and
        # softmax function
        if isinstance(self.layers[-1], SoftmaxActivation) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            # we create an object of the more efficient back propigation to be used in the backward function
            self.softmax_cross_entropy_output = ActivationSoftMaxLossCrossEntropy()
            

    # here we create a function for forward feeding our data through 
    # our network, this will be used in training and for testing
    def forward(self, X, training):

        # we forward through our input layer to get an ouput for our
        # first layer
        self.input_layer.forward(X, training = training)

        # we then loop through each layer until the last and we use the forward 
        # function with the argument taken from the previous layer variable set in 
        # finalize function
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # we then return the last layers output so it can be used for loss and accuracy
        return layer.output

    # here we will do back propigation for the network, calculating our gradients
    def backward(self, output, real_y):

        # we check if we need to use the optimised joined function
        if self.softmax_cross_entropy_output != None:

            # we start the back propigation with this optimized function
            self.softmax_cross_entropy_output.backward(output, real_y)

            # we set the softmax gradients to the joint one as it has already been calculated
            self.layers[-1].d_inputs = self.softmax_cross_entropy_output.d_inputs

            # we go through each layer from the end of the network but we do
            # not do the layer we have just done (the last activation layer)
            for layer in reversed(self.layers[:-1]):

                # we use the layers back propigation function with an argument from the next 
                # layer
                layer.backward(layer.next.d_inputs)

            # we want to end here once the loop is done
            return

        # first we do back propigation for the loss function as it is always the last
        # function in our network
        self.loss.backward(output, real_y)

        # we go through each layer from the end of the network
        for layer in reversed(self.layers):

            # we use the layers back propigation function with an argument from the next 
            # layer
            layer.backward(layer.next.d_inputs)

    def evaluate(self, val_x, val_y, *, batch_size = None):

        # default steps
        val_steps = 1

        # we check if we are using batch sizes
        if batch_size != None:
            
            # we can calculate the steps by dividing batch size and the number of
            # X input validations
            val_steps = len(val_x) // batch_size

            # if we rounded down, we add a step
            if val_steps * batch_size < len(val_x):
                val_steps += 1

        # Reset accumulated values in loss and accuracy objects 
        self.loss.new_pass()
        self.accuracy.new_pass()            

        # loop for validation steps
        for step in range(val_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_x = val_x
                batch_y = val_y

            else:
                # Otherwise slice a batch ​else​:
                batch_x = val_x[step * batch_size:(step+1) * batch_size] 
                batch_y = val_y[step * batch_size:(step+1) * batch_size]

            # we now pass our validation data through our network
            val_output = self.forward(batch_x, training = False)

            # we can now calculate the loss from our validation output
            self.loss.calculate(val_output, batch_y)

            # we now can get our validation predictions to calculate accuracy
            val_predictions = self.output_layer_activation.predictions(val_output)

            # we can now gte our accuracy for our validation predictions
            self.accuracy.calculate(val_predictions, batch_y)

        # Get validation loss and accuracy 
        val_loss = self.loss.calculate_accumulated() 
        val_accuracy = self.accuracy.calculate_accumulated()

        # now lets print our validation data
        print(
            f"validation: ",
            f"val acc: {val_accuracy:.3f}",
            f"val loss: {val_loss:.3f}"
        )

    # here we can get all the parameters of all the layers with weights and biases
    def get_params(self):

        # we initialise a params list
        params = []

        # we go through each layer with weights and bias
        for layer in self.trainable_layers:
            # we append there params to the list
            params.append(layer.get_params())

        # we now return the list
        return params

    # we can set all the parameters of all the layers with weights and biases
    def set_params(self, parameters):
        
        # we got through each set of parameters and the corrosponding layers
        for param_set, layer in zip(parameters, self.trainable_layers):
            
            # we can unpack the parameters set with '*' and pass it in to the 
            # set params function in the layer
            layer.set_params(*param_set)

    # here we can save our parameters using pythons pickle
    def save_params(self, path):
        # we are going to open to a new file called f
        with open(path, 'wb') as f:
            # we are going to 'dump' our params using pickle to this file
            pickle.dump(self.get_params(), f)

    # we should also be able to load in parameters from a file using pickle
    def load_params(self, path):

        # we first open the file passed in
        with open(path, 'rb') as f:
            # we can now use pickle to read thos parameters and we can 
            # use our set params function to save them to the model
            self.set_params(pickle.load(f))

    # here we can save entire models instead of just the parameters
    def save(self, path):

        # we need to copy the model first as we are going to edit it. this allows
        # us to save models during training as checkpoints
        model = copy.deepcopy(self)

        # lets reset our accumulated batch loss and accuracy
        model.loss.new_pass()
        model.accuracy.new_pass()

        # next we can remove any data in the input layer and reset our gradients
        # __dict__ lists all the names and values in our instances properties, we
        # use pop to pop out the given name, we use false afterwards to prevent
        # any errors from appearing if these names dont exist in the objects properties
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('d_inputs', None)

        # lets oterate over all the layers to remove their properties
        for layer in model.layers:
            # we loop over each property in the layers we want to remove
            for property in ['inputs', 'output', 'd_inputs', 'd_weights', 'd_bias']:
                # we use the same method as above to remove them
                layer.__dict__.pop(property, None)

        # we can now save the model using pickle like how we saved the parameters
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # now we need a method for loading our model, we can use a static method decorator
    # which allows us to call this method when the model isnt initialised. 
    # ( e.g. model = Model.load_model() )
    @staticmethod
    def load_model(path):
        
        # we first open the file passed in
        with open(path, 'rb') as f :
            # we then use pickle to load the file
            model = pickle.load(f)
        
        # we can now return our model
        return model

#---------- use case -----------#

'''# train data
X, y = spiral_data(samples = 1000, classes=3)

# test data
X_test, y_test = spiral_data(samples = 1000, classes = 3)

# creating our model
model = Model()

# adding our layer
model.add(DenseLayer(2, 512, weight_reg_l2 = 5e-4, bias_reg_l2 = 5e-4))
model.add(ReLUActivation())
model.add(DropoutLayer(0.1))
model.add(DenseLayer(512, 3))
model.add(SoftmaxActivation())

# setting our optimizer and loss functions
model.set(
    loss = CategoricalCrossEntropyLoss(), 
    optimizer = AdamOptimizer(learning_rate = 0.05, decay_rate=5e-5),
    accuracy = CategoricalAccuracy()
)

# we finalise our model (setup for training)
model.finalize()

# training our model
model.train(X, y, validation_data=(X_test,y_test), epochs = 10000, print_every = 100)

'''