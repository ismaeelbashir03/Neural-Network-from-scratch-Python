import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from NN import Model, DenseLayer, CategoricalAccuracy, SoftmaxActivation, ReLUActivation, AdamOptimizer, CategoricalCrossEntropyLoss

def load_mnist(dataset, path):
    # scanning all directories to create a label list
    labels = os.listdir(os.path.join(path, dataset))

    # creating a list for samples and labels
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file ), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def create_data(path):
    X, y = load_mnist('train',path)
    X_test, y_test = load_mnist('test', path)

    return X, y, X_test, y_test

X, y, X_test, y_test = create_data('fashion_mnist_images')

# we can now shuffle our data, we will do this by using keys, 
# which hold the indexes of our data, which will be shuffled then applies
# to our data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale features to be between -1 and 1 and we also
# flatten our images so they can fit into our dense layers (1d input), some CNN's 
# can take 2d arrays but even then most flatten them before use
# we put -1 in the second argument as we still want the 60k different images to be 
# seperated and not flattened together
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

'''# we create our model
model = Model()

# we add our layers
model.add(DenseLayer(X.shape[1], 128))
model.add(ReLUActivation())
model.add(DenseLayer(128, 128))
model.add(ReLUActivation())
model.add(DenseLayer(128, 10))
model.add(SoftmaxActivation())

# setting our loss, optimizationa and accuracy
model.set(
    loss = CategoricalCrossEntropyLoss(),
    optimizer = AdamOptimizer(decay_rate=1e-3),
    accuracy = CategoricalAccuracy()
)

# finalising our model
model.finalize()

# training our model
model.train(X, 
            y, 
            validation_data = (X_test, y_test), 
            epochs = 10,
            batch_size = 128,
            print_every = 100
)

# evaluating our model, 
# (we are using the validation data so it will be the same output as before)
model.evaluate(X_test, y_test)

# saving our parameters
params = model.get_params()

# creating a new model
model2 = Model()

# adding layers to our new model that match our parameters
model2.add(DenseLayer(X.shape[1], 128))
model2.add(ReLUActivation())
model2.add(DenseLayer(128, 128))
model2.add(ReLUActivation())
model2.add(DenseLayer(128, 10))
model2.add(SoftmaxActivation())

# setting our accuracy and loss functions
model2.set(
    loss = CategoricalCrossEntropyLoss(),
    accuracy = CategoricalAccuracy()
)

# finalising our model
model2.finalize()

# loading in our model 1 parameters
model2.set_params(params)

# we can now evaluate on our loaded model
model2.evaluate(X_test, y_test)

# here we save our model to a params file to be loaded in later or in another file
model2.save_params('fashion_params.params')

# here we are loading in our params file to be used in model 1
model.load_params('fashion_params.params')

# we can now evaluate on our loaded model
model.evaluate(X_test, y_test)

# lets save our model 1 to a model file
model.save('model1.model')
'''
# lets load model 1 to a new model, model 3
model3 = Model.load_model('model1.model')

model3.evaluate(X_test, y_test)

# we can now shuffle our test data, we will do this by using keys, 
# which hold the indexes of our data, which will be shuffled then applies
# to our data
keys = np.array(range(X_test.shape[0]))
np.random.shuffle(keys)
X_test = X_test[keys]
y_test = y_test[keys]

# lets get the confidences with model 3 on the first 5 test samples
confidences = model3.predict(X_test[5:11])

# we can now use our activation functions prediction method which will transform these
# values in pure predictions (e.g. [0,1,0], [1,0,0] -> [1,0], we get the index of what the
# model thinks is the piece of clothing)
predictions = model3.output_layer_activation.predictions(confidences)

# making a dictionary for our clothing items
fashion_MNIST_labels = {
    0: 'T-shirt/top', 
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat', 
    5: 'Sandal', 
    6: 'Shirt', 
    7: 'Sneaker', 
    8: 'Bag',
    9: 'Ankle boot'
}

# looping through our predictions and using the index to print the clothing item
for pred in predictions:
    print("Model predicted: " + fashion_MNIST_labels[pred])

# doing the same for the actual data
for actual in y_test[5:11]:
    print("actual item: " + fashion_MNIST_labels[actual])

# now lets try hand drawn images, given by nnfs

# we read in the imagez with cv2, grayscale as our model only does black and white
# (i.e. values between 0 and 255)
t_shirt_image = cv2.imread('t_shirt.png', cv2.IMREAD_GRAYSCALE)
t_shirt_copy = t_shirt_image.copy()
trouser_image = cv2.imread('trouser.png', cv2.IMREAD_GRAYSCALE)
trouser_copy = trouser_image.copy()

# we need to resize the images to 28 by 28 pixels fit in our model
t_shirt_image = cv2.resize(t_shirt_image, (28, 28))
trouser_image = cv2.resize(trouser_image, (28, 28))

# we neeed to inverse the colors as our training data had black backgrounds, but this does not
# we need to resize the images to 28 by 28 pixels fit in our model
# we have to do this as we are not using a convolutional neural network, which learns features
# like curves our line, this network only learns values, so it learns contrast instead.
t_shirt_image = 255 - t_shirt_image 
trouser_image = 255 - trouser_image

# we next need to flatten and scale down the data to be netween -1 and 1
t_shirt_image = (t_shirt_image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
trouser_image = (trouser_image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# now we can feed this data into our model
confidences = model3.predict(t_shirt_image)

# we can get the pure predictions
predictions = model3.output_layer_activation.predictions(confidences)

# we show the images with the models prediction
plt.imshow(t_shirt_copy, cmap = 'gray')
plt.text(5, 10, f"Model predicted: {fashion_MNIST_labels[predictions[0]]}")
plt.text(5, 20, "Actual: T-Shirt")
plt.savefig('T-shirt_prediction')
plt.show()

# now we can feed the next data into our model
confidences = model3.predict(trouser_image)

# we can get the pure predictions
predictions = model3.output_layer_activation.predictions(confidences)

# we show the image with the models prediction
plt.imshow(trouser_copy, cmap = 'gray')
plt.text(5, 10, f"Model predicted: {fashion_MNIST_labels[predictions[0]]}")
plt.text(5, 20, "Actual: Trouser")
plt.savefig("Trouser prediction")
plt.show()

"""
validation:  val acc: 0.870 val loss: 0.359
Model predicted: Bag
Model predicted: Sneaker
Model predicted: Sneaker
Model predicted: Trouser
Model predicted: Pullover
Model predicted: Shirt
actual item: Bag
actual item: Sneaker
actual item: Sneaker
actual item: Trouser
actual item: Pullover
actual item: Shirt
"""