import numpy as np
!pip install autopep8

class Neuron:
    def __init__(self, input_size, learning_rate=0.1):    
        self.lr = learning_rate
        self.W = np.random.uniform(-1, 1, input_size)
        self.b = np.random.randn() 
        self.X = None
        self.ypred = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  

    def forward(self, X):
        self.X = X
        z = np.dot(self.W, self.X) + self.b
        self.ypred = self.sigmoid(z)
        return self.ypred

    def backward(self, ytrue):
        d_loss = 2 * (self.ypred - ytrue)
        d_sigmoid = self.ypred * (1 - self.ypred)
        gradient_z = d_loss * d_sigmoid
        gradient_w = gradient_z * self.X
        gradient_b = gradient_z

        self.W = self.W - self.lr * gradient_w
        self.b = self.b - self.lr * gradient_b

        loss = (self.ypred - ytrue) ** 2
        return loss

if __name__ == "__main__":
   
    X_input = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    target = 1.0 
    my_neuron = Neuron(input_size=25, learning_rate=0.5)
    print("Start Learning")
    

    for epoch in range(1, 11):
        prediction = my_neuron.forward(X_input)
        current_loss = my_neuron.backward(target)
        print(f"Epoch {epoch:2d} | Prediction: {prediction:.5f} | Loss: {current_loss:.5f}")


