import numpy as np 
import matplotlib.pyplot as plt 

class SinglePerceptronTrainer:
    def __init__(self):
        self.weight =  [0,0,0] 
        self.feature = [[0,1,0],[0,0,1],[0,1,1]]
        self.dotproducts = [0,0,0]
        self.target = [1,1,0]
        self.estimates = [0,0,0]
        self.bias = 0.1
        self.rate = 0.1
        self.iterations = []

    def weighted_sum(self):
        dotproduct = [0 for _ in self.feature]
        for i in range(len(self.feature)):
            for j in range(len(self.feature[i])):
                dotproduct[i] += self.feature[i][j] * self.weight[j]  # Calculate weighted sum
            dotproduct[i] += self.bias  
            

        self.dotproducts = dotproduct  # Update class variable
        return dotproduct

    #steg-funktion. 
    # def activation_function(self, dotproduct:list):
    #     #detta är en steg-funktion, ej sigmoid, denna är binär. den kan endast säga 1 eller 0. 
    #     outputs = []
    #     for x in dotproduct:
    #         if x > 0:
    #             outputs.append(1)
    #         else:
    #             outputs.append(0)
    #     return outputs
    
    #Sigmoid funktion. 
    def activation_function(self,z):
        outputs = []
        for x in z: 
            output = 1/(1 + np.exp(-x))
            outputs.append(output)
        return outputs

    def loss_calculation(self, outputs:list):
        losses = []
        #processar alla outputs från activation.
        for i in range(len(outputs)):
            l = 0.5 * (self.target[i] - outputs[i])**2
            losses.append(l)
        return losses
    
    def train(self):
        '''
        
        sätt self.iterations och range till samma.
        
        '''

        self.iterations = list(range(1,2000))
        self.all_losses = []
        self.all_outputs = []

        for x in range(1, 2000):
            self.weighted_sum()
            outputs = self.activation_function(self.dotproducts)
            losses = self.loss_calculation(outputs)
            
            self.all_outputs.append(outputs)  # Store outputs for visualization
            self.all_losses.append(sum(losses) / len(losses))  # Store average loss for visualization
            for i in range(len(outputs)):
                error = self.target[i] - outputs[i]
                for j in range(len(self.weight)):
                    self.weight[j] += self.rate * error * self.feature[i][j]
                self.bias += self.rate * error

        return self.iterations, self.all_losses, self.all_outputs #returnar bara för visualisation

    def test(self, new_feature):
        z = (new_feature*self.weight) + self.bias
        prediction = self.activation_function(z)
        return prediction

ai = SinglePerceptronTrainer()

iterations, losses, outputs = ai.train()

for x in ai.all_outputs:
    print(x)

#dessa figurer visar att förlusten minskar ju fler iterationer modellen tränas. 
plt.figure(figsize=(10,5))
plt.plot(iterations, losses, label='Loss', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.title('Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Denna visar att estimatet kommer närare target ju fler iterationer. 
plt.figure(figsize=(10,5))
plt.plot(iterations, outputs, label='Estimate', color='red')
plt.xlabel('Iterations')
plt.ylabel('Estimate Value')
plt.title('Estimate over Iterations')
plt.legend()
plt.grid(True)
plt.show()