import numpy as np 
import matplotlib.pyplot as plt 

class SinglePerceptronTrainer:
    def __init__(self):
        self.weight = np.random.randn() * 0.01  # Initiate with a small random value close to 0
        self.target = 1
        self.feature = 0.2
        self.bias = 0
        self.rate = 0.1
        self.iterations = []

    def weighted_sum(self):
        z = (self.feature*self.weight) + self.bias 
        return z
    
    # def activation_function(self, z):
    #     detta är en steg-funktion den är binär. den kan endast säga ja eller nej dvs. 
    #     if z > 0:
    #         return 1
    #     else:
    #         return 0
    
    
    def activation_function(self,z):
        #denna är en sigmoid activation function, den är bättre för att ge icke-binära resultat
        a = 1/(1 + np.exp(-z))
        return a

    def loss_calculation(self):
        x = self.activation_function(self.weighted_sum())
        l = 0.5*(self.target - x)**2
        return l 
    
    #backward propagation används i neurala nätverk, inte single perceptron
    #men dlda, dadz, dzdw, dldw, dldb, är deriveringar som isåfall skulle behövas. 
    # def backward_propagation(self):
    #     a = self.activation_function()
    #     dlda = 2*(self.target-a)
    #     dadz = a * (1-a)
    #     dzdw = self.feature
    #     dldw = dlda * dadz * dzdw 
    #     dldb = dlda*dadz 
    #     self.bias = self.bias - self.rate * dldb
    #     self.weight = self.weight - self.rate * (dldw)
    #     return 
    
    def train(self):
        #dessa 3 listor är till för att kunna visualisera resultatet i pyplot (grafer)
        self.iterations = list(range(0,100))
        losses = []
        estimates = []

        for x in range(0,100):
            z = self.weighted_sum()
            y = self.activation_function(z)
            self.weight = self.weight + self.rate * (self.target - y)*self.feature
            self.bias = self.bias + self.rate * (self.target - y)
            losses.append(self.loss_calculation())
            estimates.append(y)
        
        #returna losses och estimates för visualiseringen 
        return self.loss_calculation(), self.activation_function(self.weighted_sum()), losses, estimates

    def test(self, new_feature):
        z = (new_feature*self.weight) + self.bias
        prediction = self.activation_function(z)
        return prediction

ai = SinglePerceptronTrainer()

loss, estimate, losses, estimations = ai.train()

#print ai.test(0.3) kommer ge ungefär 0.9. så om vi säger att 0.3 är en persons längd, så kommer 0.9 vara personens vikt. 
#modellen har tränats på en person som är 0.2 i höjd och väger 1. så det stämmer bra överens att 0.2 bör ge 0.9. 
print(ai.test(0.3))

#dessa figurer visar att förlusten minskar ju fler iterationer modellen tränas. 
plt.figure(figsize=(10,5))
plt.plot(ai.iterations, losses, label='Loss', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.title('Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Denna visar att estimatet ökar (kommer närare target) ju fler iterationer. 
plt.figure(figsize=(10,5))
plt.plot(ai.iterations, estimations, label='Estimate', color='red')
plt.xlabel('Iterations')
plt.ylabel('Estimate Value')
plt.title('Estimate over Iterations')
plt.legend()
plt.grid(True)
plt.show()
