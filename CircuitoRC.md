---
layout: default
title: Customization
nav_order: 2
---

# Circuito RLC 

Propagación de error para Frecuencia de resonancia


```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
```


```python
init_printing(use_unicode=True)

```


```python
L, C, e_C, e_L = symbols('L C e_C e_L')
fr = 1/(2*pi)*1/sqrt(L*C)
fr.evalf(subs={C:10E-9, L:10E-3})
print('La frecuencia será %.0f Hz' %fr.evalf(subs={C:10E-9, L:10E-3}))
```

    La frecuencia será 15915 Hz
    


```python
dfr = sqrt(simplify(diff(fr, C))**2*e_C**2 + simplify(diff(fr, L))**2*e_L**2)
dfr
```




$\displaystyle \sqrt{\frac{e_{L}^{2}}{16 \pi^{2} C L^{3}} + \frac{e_{C}^{2}}{16 \pi^{2} C^{3} L}}$




```python
r = dfr.subs([(C, 10.0E-9), (L, 10.0E-3), (e_C, (10.0E-9*0.08)), (e_L, (10.0E-3*0.05))])
r.evalf()
```




$\displaystyle 750.732365101242$


