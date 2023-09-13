"""
Ths module contains the reaction model functions $f(\alpha)$
"""

## Power Laws

### P4: $ f\left(\alpha\right) = 4 \alpha^{3/4} $

def P4(a,integral = False):
    """
    Power Law (P4) model
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return a**(1/4)
    else:
        return 4*(a**(3/4))

### P3: $f\left(\alpha\right) = 3\alpha^{2/3}$

def P3(a,integral = False):
    """
    Power Law (P3) model
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return a**(1/3)
    else:
        return 3*(a**(2/3))

### P2: $f\left(\alpha\right) = 2\alpha^{1/2}$

def P2(a, integral = False):
    """
    Power Law (P2) model
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return a**(1/2)
    else:
        return 2*(a**(1/2))

### P2/3: $f\left(\alpha\right) = \frac{2}{3}\alpha^{-1/2}$

def P2_3(a, integral = False):
    """
    Power Law (P2/3) model
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return a**(3/2)
    else:
        return (2/3)*(a**(-1/2))

## Diffusion

### One dimensional D1: $f\left(\alpha\right) = \frac{1}{2}\alpha^{-1}$

def D1(a, integral = False):
    """
    One dimensional diffusion model (D1)
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return a**2
    else:
        return (1/2)*a**(-1)

### Two dimensional D2: $f\left(\alpha\right) = \left[-\ln{\left(1-\alpha\right)}\right]^{-1}$

def D2(a, integral = False):
    """
    Two dimensional diffusion model (D2)
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return ((1-a)*np.log(1-a)) + a
    else:
        return 1/((-1)*np.log(1-a))

### Three dimensional D3: $f\left(\alpha\right) = \frac{3}{2}\left(1-\alpha\right)^{2/3}\left[1-\left(1-\alpha\right)^{1/3}\right]^{-1}$

def D3(a, integral = False):
    """
    Three dimensional diffusion model (D3)
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return (1-((1-a)**(1/3)))**2
    else:
        return (3/2)*((1-a)**(2/3))*(1/(1-((1-a)**(1/3))))

## Mampel (F1): $f\left(\alpha\right) = 1-\alpha$ 

def F1(a, integral = False):
    """
    Mampel (First order) model (F1)
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return -(np.log(1-a))
    else:
        return 1-a

## Avrami-Erofeev

### A4: $f\left(\alpha\right) = 4\left(1-\alpha\right)\left[\ln{\left(1-\alpha\right)}\right]^{3/4}$

def A4(a, integral = False):
    """
    Avrami-Erofeev (A4) model 
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return (-np.log(1-a))**(1/4)
    else:
        return 4*(1-a)*((-np.log(1-a))**(3/4))

### A3: $f\left(\alpha\right) = 3\left(1-\alpha\right)\left[\ln{\left(1-\alpha\right)}\right]^{2/3}$

def A3(a, integral = False):
    """
    Avrami-Erofeev (A3) model 
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return (-np.log(1-a))**(1/3)
    else:
        return 3*(1-a)*((-np.log(1-a))**(2/3))

### A2: $f\left(\alpha\right) = 2\left(1-\alpha\right)\left[\ln{\left(1-\alpha\right)}\right]^{1/2}$

def A2(a, integral = False):
    """
    Avrami-Erofeev (A2) model 
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        return (-np.log(1-a))**(1/2)
    else:
        return 2*(1-a)*((-np.log(1-a))**(1/2))

## Contractions

### Contracting sphere (R3): $f\left(\alpha\right) = 3\left(1-\alpha\right)^{2/3}$

def R3(a, integral =  False):
    """
    Contracting sphere (R3) model 
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        1 - ((1-a)**(1/3))
    else:
        return 3*((1-a)**(2/3))

### Contracting cylinder (R2): $f\left(\alpha\right) = 2\left(1-\alpha\right)^{1/2}$

def R2(a, integral =  False):
    """
    Contracting cylinder (R2) model 
    
    Parameters:   a : (\alpha) Conversion degree value.
    
    Returns:    f(a): Reaction model evaluated on the conversion degree
    """
    if integral == True:
        1 - ((1-a)**(1/2))
    else:
        return 2*((1-a)**(1/2))
