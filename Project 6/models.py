import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        result = nn.as_scalar(self.run(x))
        if result >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        batch_size = 1
        change_flag = True
        while change_flag:
            change_flag = False
            for x, y in dataset.iterate_once(batch_size):
                result = self.get_prediction(x)
                if result != nn.as_scalar(y):
                    self.w.update(nn.Constant(nn.as_scalar(y) * x.data), 1)
                    change_flag = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.input_dim = 1
        self.hidden_layer_size = 512
        self.batch_size = 200
        self.learning_rate = -0.05
        self.w1 = nn.Parameter(self.input_dim, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        return predicted_y


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SquareLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        import time
        tick = time.time()
        for x, y in dataset.iterate_forever(self.batch_size):
            # print("Training!!:")
            grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = \
                nn.gradients(self.get_loss(x, y), [self.w1, self.b1, self.w2, self.b2])
            self.w1.update(grad_wrt_w1, self.learning_rate)
            self.b1.update(grad_wrt_b1, self.learning_rate)
            self.w2.update(grad_wrt_w2, self.learning_rate)
            self.b2.update(grad_wrt_b2, self.learning_rate)
            # print(str(nn.as_scalar(self.get_loss(x, y))))
            if nn.as_scalar(self.get_loss(x, y)) < 0.015:
                # print(str(nn.as_scalar(self.get_loss(x, y))))
                tock = time.time()
                durationSeconds = tock - tick
                print(str(nn.as_scalar(self.get_loss(x, y))))
                print(durationSeconds)
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.input_dim = 784
        self.output_dim = 10
        self.hidden_layer_size = 200
        self.batch_size = 100
        self.learning_rate = -0.5
        self.w1 = nn.Parameter(self.input_dim, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, self.output_dim)
        self.b2 = nn.Parameter(1, self.output_dim)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        return predicted_y


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        import time
        tick = time.time()
        for x, y in dataset.iterate_forever(self.batch_size):
            # print("Training!!:")
            grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = \
                nn.gradients(self.get_loss(x, y), [self.w1, self.b1, self.w2, self.b2])
            self.w1.update(grad_wrt_w1, self.learning_rate)
            self.b1.update(grad_wrt_b1, self.learning_rate)
            self.w2.update(grad_wrt_w2, self.learning_rate)
            self.b2.update(grad_wrt_b2, self.learning_rate)
            # print(str(nn.as_scalar(self.get_loss(x, y))))
            # print(dataset.get_validation_accuracy())
            # print(time.time()-tick)
            if dataset.get_validation_accuracy() > 0.975:
                tock = time.time()
                durationSeconds = tock - tick
                print(str(nn.as_scalar(self.get_loss(x, y))))
                print(durationSeconds)
                return
            # if nn.as_scalar(self.get_loss(x, y)) < 0.015:
            #     # print(str(nn.as_scalar(self.get_loss(x, y))))
            #
            #     return


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.input_dim = self.num_chars
        self.output_dim = len(self.languages)
        self.hidden_layer_size = 500
        self.batch_size = 100
        self.learning_rate = -0.4
        self.w1 = nn.Parameter(self.input_dim, self.hidden_layer_size)
        self.w1_hidden = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.b1_hidden = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, self.output_dim)
        self.b2 = nn.Parameter(1, self.output_dim)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        h_i = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w1), self.b1))
        for i in range(1, len(xs)):
            h_ii = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(h_i, self.w1_hidden), nn.AddBias(nn.Linear(xs[i], self.w1),
                                                                                        self.b1)), self.b1_hidden))
            h_i = h_ii
        h1 = h_i
        predicted_y = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        return predicted_y


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        import time
        tick = time.time()
        for xs, y in dataset.iterate_forever(self.batch_size):
            # print("Training!!:")
            grad_wrt_w1, grad_wrt_w1_hidden, grad_wrt_b1, grad_wrt_b1_hidden, grad_wrt_w2, grad_wrt_b2 = \
                nn.gradients(self.get_loss(xs, y), [self.w1, self.w1_hidden, self.b1, self.b1_hidden, self.w2, self.b2])
            self.w1.update(grad_wrt_w1, self.learning_rate)
            self.w1_hidden.update(grad_wrt_w1_hidden, self.learning_rate)
            self.b1.update(grad_wrt_b1, self.learning_rate)
            self.b1_hidden.update(grad_wrt_b1_hidden, self.learning_rate)
            self.w2.update(grad_wrt_w2, self.learning_rate)
            self.b2.update(grad_wrt_b2, self.learning_rate)

            # print(dataset.get_validation_accuracy())
            # print(time.time()-tick)
            if dataset.get_validation_accuracy() > 0.9:
                tock = time.time()
                durationSeconds = tock - tick
                # print(str(nn.as_scalar(self.get_loss(xs, y))))
                print(durationSeconds)
                return
            # if nn.as_scalar(self.get_loss(x, y)) < 0.015:
            #     # print(str(nn.as_scalar(self.get_loss(x, y))))
            #
            #     return