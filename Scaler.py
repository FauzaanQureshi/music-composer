class Scaler:
    def __init__(self, n):
        self.MAX = n.max()
        self.MIN = n.min()

    def scale(self, n):
        return (self.MAX-n)/(self.MAX-self.MIN)
    
    def descale(self, x):
        return self.MAX-(self.MAX-self.MIN)*x