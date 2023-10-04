class SinglePerceptron:
    def __init__(self):
        self.weights = [0.0,0.0,0.0]
        self.feautures = [[0.0,1.0,0.0], [0.0,0.0,1.0], [0.0,1.0,1.0]]
        self.output = 0.0 #w0
        self.output1 = 0.0 #w1
        self.output2 = 0.0 #w2
        self.dotproduct = 0.0
        self.threshold = 0
        self.estimate = [0,0,0]
        self.target = [1,1,0]
        self.learningrate = 0.1

    def dotproduction(self):
        for x in self.feautures:
            for y in x:
                if x == 0:
                    self.output += self.weights[0] * y 
                elif x == 1:
                    self.output1 += self.weights[1] * y
                elif x == 2:
                    self.output2 += self.weights[2] * y
            self.dotproduct = self.output + self.output1 + self.output2

    def compare_against_threshold(self):
        for i in self.estimate:
            if self.dotproduct > self.threshold:
                self.estimate[i] = 1
            else:
                self.estimate[i] = 0
    
    def backward_propagation(self):            
        for i, weight in enumerate(self.weights):
            for z in self.feautures[i]:
                self.weights[i] = weight + self.learningrate * (self.target[i] - self.estimate[i]) * z      
            print(self.target[i], self.estimate[i], end=" ")
                          #0,1*(1-0)*0
            #print(self.weights[i])
        


d1 = SinglePerceptron()
d1.dotproduction()
d1.backward_propagation()
d1.dotproduction()
d1.compare_against_threshold()
print(d1.estimate)
            
