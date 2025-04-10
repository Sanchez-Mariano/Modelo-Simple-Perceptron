import numpy as np
import matplotlib.pyplot as plt

[ ] #datos para entrenar el modelo "Perceptrón"

np.random.seed(42)

x = np.random.rand(20)
y = 2*x + (np.random.rand(20)-0.5)*0.5

plt.plot(x,y,"b.")
plt.grid(True)
plt.show()

# funciones necesarias para entrenar el perceptrón

def gradient(w, x, y): 
    dldw = x*w - y
    dydw = x
    dldw = dldw*dydw
    return np.mean(2*dldw)

def cost(y, y_hat): 
    return ((y_hat - y)**2).mean()

def solve(epochs = 29, w = 1.2, lr = 0.2):
    # se itera un número determinado de `epochs`, por cada uno, se calcula gradientes y se actualizan los pesos
    weights = [(w, gradient(w, x, y), cost(x*w, y))]
    for i in range(1,epochs+1):
        dw = gradient(w, x, y)
        w = w - lr*dw
        weights.append((w, dw, cost(x*w, y)))
    return weights


from matplotlib import animation, rc
rc('animation', html='html5')

y_min = y.min() - 0.1
y_max = y.max() + 0.1

def update(i):
    xs = np.linspace(0, 1, num=100) 
    (w, dw, cost_val) = weights[i]
    ax.clear()
    ax.plot(x, y, "b.")
    ax.plot(xs, w*xs, "-k")
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$y$", rotation=0, fontsize=14)
    ax.set_title(f"Iteración: {i}, Costo: {cost_val:.4f}")
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(y_min, y_max) 
    return ax
 
weights = solve()
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(1,1,1)
anim = animation.FuncAnimation(fig, update, frames=len(weights), interval=100)
plt.close()

anim.save("animation.gif")