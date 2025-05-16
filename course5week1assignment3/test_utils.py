from termcolor import colored
from keras.src.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.src.layers import concatenate, ZeroPadding2D, Dense, LSTM, RepeatVector

def comparator (learner, instructor):
    if (len(learner) != len(instructor)):
        raise AssertionError("error in test, diff number of elements")
    for index, a in enumerate(instructor):
        b = learner[index]
        if tuple(a) != tuple(b):
            print(colored(f"Test failed at index {index}", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"),
                  "\n\n does not match the input value: \n\n",
                  colored(f"{a}", "red"))
            raise AssertionError("error in test")
    print(colored("all tests passed", "green"))

def summary(model):
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
        if (type(layer) == ZeroPadding2D):
            descriptors.append(layer.padding)
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
        if (type(layer) == LSTM):
            descriptors.append(layer.input_shape)
            descriptors.append(layer.activation.__name__)
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
        result.append(descriptors)
    return result

