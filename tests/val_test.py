import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grad_engine.value import value
from grad_engine.value import log

a=value(10)

c=log(a)

c.backward()

print(a)
print(c)