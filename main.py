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

    def weighted_sum(self, features:list):
        features_copy = features.copy()
        dotproduct = [0 for _ in features_copy]
        for i in range(len(features_copy)):
            for j in range(len(features_copy[i])):
                dotproduct[i] += features_copy[i][j] * self.weight[j]
            dotproduct[i] += self.bias
        if modify_internal_state:
            self.dotproducts = dotproduct
        return dotproduct


    #steg-funktion. 
    def activation_function(self, dotproduct:list):
        #detta är en steg-funktion, ej sigmoid, denna är binär. den kan endast säga 1 eller 0. 
        outputs = []
        for x in dotproduct:
            if x > 0:
                outputs.append(1)
            else:
                outputs.append(0)
        return outputs
    
    #Sigmoid funktion. 
    # def activation_function(self,z):
    #     outputs = []
    #     for x in z: 
    #         output = 1/(1 + np.exp(-x))
    #         outputs.append(output)
    #     return outputs

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

        self.iterations = list(range(1,20))
        self.all_losses = []
        self.all_outputs = []

        for x in range(1, 20):
            self.weighted_sum(self.feature)
            outputs = self.activation_function(self.dotproducts)
            losses = self.loss_calculation(outputs)
            
            self.all_outputs.append(outputs)  # Store outputs for visualization (plot will show 3 lines)
            self.all_losses.append(sum(losses) / len(losses))  # Store average loss for visualization (loss shows 1 line representing all 3 estimates)
            for i in range(len(outputs)):
                error = self.target[i] - outputs[i]
                for j in range(len(self.weight)):
                    self.weight[j] += self.rate * error * self.feature[i][j]
                self.bias += self.rate * error

        return self.iterations, self.all_losses, self.all_outputs 

    # def test(self, new_feature):
    #     dotproducts = self.weighted_sum(new_feature, modify_internal_state=False)
    #     prediction = self.activation_function(dotproducts)
    #     return prediction

ai = SinglePerceptronTrainer()

iterations, losses, outputs = ai.train()

#printa estimates. 
for x in ai.all_outputs:
    print(f"train:",x)

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