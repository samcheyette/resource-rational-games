import random
import numpy as np
import scipy.stats as st
from math import log, exp
from itertools import combinations, product
import copy




class Number:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"{repr(self.value)}"
    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value
    def execute(self):
        return self.value


class Float(Number):
    def __eq__(self, other):
        return isinstance(other, Float) and self.value == other.value


class Int(Number):
    def __eq__(self, other):
        return isinstance(other, Int) and self.value == other.value


class Flip:
    def __init__(self, p=Number(0.5), arg1=Number(0), arg2 = Number(1)):
        self.p = p  
        self.arg1 = arg1
        self.arg2 = arg2

    def __repr__(self):
        if self.p.execute() == 0.5:
            return "Flip()"
        else:
            return f"Flip({repr(self.p)})"

    def __eq__(self, other):
        return (isinstance(other, Flip) and self.p == other.p and 
                    other.arg1 == self.arg1 and other.arg2 == self.arg2)

    def execute(self):
        rand = random.random()
        if rand < self.p.execute():
            return self.arg1.execute()
        else:
            return self.arg2.execute()



class If:
    def __init__(self, condition, x, y):
        self.condition = condition  
        self.x = x
        self.y = y

    def __repr__(self):
        return f"If({repr(self.condition)}, {repr(self.x)}, {repr(self.y)})"

    def __eq__(self, other):
        return (isinstance(other, If) and self.condition == other.condition and 
                    other.x == self.x and other.y == self.y)

    def execute(self):
        if self.condition.execute():
            return self.x.execute()
        else:
            return self.y.execute()


class List:
    def __init__(self, *elements):
        self.elements = list(elements)

    def __repr__(self):
        return f"{repr(self.elements)}"

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return isinstance(other, List) and self.elements == other.elements


    def __iter__(self):
        return iter(self.elements)

    def execute(self):
        return [el.execute() for el in self.elements]



class Dirichlet:
    def __init__(self, *elements):
        self.elements = list(elements)


    def __repr__(self):
        #return f"{[round(el.item(),2) for el in self.elements]}"
        return f"{[round(el.item(),2) for el in self.elements]}"



    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return isinstance(other, Dirichlet) and self.elements == other.elements


    def execute(self):
        return [el for el in self.elements]



class Categorical:
    def __init__(self, ps):
        self.ps = ps
    def __repr__(self):
        return f"Categorical({repr(self.ps)})"

    def __eq__(self, other):
        return (isinstance(other, Categorical) and self.ps == other.ps)

    def execute(self):
        return random.choices([i for i in range(1,len(self.ps)+1)], weights=self.ps.execute())[0]
        



        

class Tuple:
    def __init__(self, *elements):
        self.elements = elements

    def __repr__(self):
        return f"{repr(self.elements)}"

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return isinstance(other, Tuple) and self.elements == other.elements


    def __iter__(self):
        return iter(self.elements)

    def execute(self):
        return tuple([el.execute() for el in self.elements])





class Repeat:
    def __init__(self, element, n_repeat):
        self.element = element
        self.n_repeat = n_repeat

    def __repr__(self):
        return f"Repeat({repr(self.element)}, {repr(self.n_repeat)})"


    def __eq__(self, other):
        return isinstance(other, Repeat) and self.element == other.element and self.n_repeat == other.n_repeat


    def execute(self):
        return tuple([self.element.execute() for _ in range(self.n_repeat.execute())])
