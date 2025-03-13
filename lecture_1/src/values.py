import random

class Values:
    def __init__(self):
        self.randomArray = [random.randint(1, 10) for _ in range(10)]
        self.data = {
            "Index": [1, 2, 3, 4],
            "Feature 1": [1.4, 1.7, 4.7, 4.0],
            "Feature 2": [0.2, 0.2, 1.4, 1.3]
        }
        self.w = [-0.708, -0.338]
        self.b = 2.271
