import torch
import matplotlib
import matplotlib.pyplot as plt
from xitorch.integrate import solve_ivp

matplotlib.use('TkAgg')

def euler_forward(fcn, ts, y0, params, verbose=False, **unused):
    with torch.no_grad():
        yt = torch.empty((len(ts), *y0.shape), dtype=y0.dtype, device=y0.device)
        yt[0] = y0
        for i in range(len(ts)-1):
            yt[i+1] = yt[i] + (ts[i+1] - ts[i]) * fcn(ts[i], yt[i], *params)
        if verbose:
            print("Done")
        return yt

def main():
    fcn = lambda t, y, a: -a * y
    ts = torch.linspace(0, 2, 1000, requires_grad=True)
    a = torch.tensor(1.2, requires_grad=True)
    y0 = torch.tensor(1.0, requires_grad=True)
    yt = solve_ivp(fcn, ts, y0, params=(a,), method=euler_forward)  # custom implementation
    _ = plt.plot(ts.detach(), yt.detach())  # y(t) = exp(-a*t)
    plt.title("Euler Forward")
    plt.show()

    # first order grad
    grad_a, = torch.autograd.grad(yt[-1], a, create_graph=True)
    grad_a_true = -ts[-1] * torch.exp(-a*ts[-1])  # dy/da = -t*exp(-a*t)
    print(grad_a.data, grad_a_true.data)

    # second order grad
    grad_a2, = torch.autograd.grad(grad_a, a)
    grad_a2_true = ts[-1]**2 * torch.exp(-a*ts[-1])  # d2y/da2 = t*t*exp(-a*t)
    print(grad_a2.data, grad_a2_true.data)

if __name__ == "__main__":
    main()