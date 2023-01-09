import cv2
import os
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
# which hold the indexes of our data, whihc will be shuffled then applies
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

# we create our model
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

# lets load model 1 to a new model, model 3
model3 = Model.load_model('model1.model')

model3.evaluate(X_test, y_test)
"""
step 468 acc: 0.875 loss: 0.292 data loss: 0.292 reg loss: 0.000 learn rate: 0.00017577781683951485
training,  acc: 0.892,  loss: 0.298 ( data_loss: 0.298,  reg_loss: 0.000),  lr: 0.00017577781683951485
validation:  val acc: 0.870 val loss: 0.359
"""