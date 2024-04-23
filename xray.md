---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python tags=["initialize"]
import numpy as np

from math import pi
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import matplotlib.image as img

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from common import draw_classic_axes, configure_plotting

configure_plotting()
```

_based on chapters 13.1-13.2 & 14.1-14.2 of the book_  

!!! success "Expected prior knowledge"

    Before the start of this lecture, you should be able to:

    - describe crystal structures using crystallographic terminology (lattice, basis, unit cells, etc. as introduced in the previous lecture) 
    - recall that waves will interfere constructively or destructively depending on their relative phase
    - describe the basic concepts of the reciprocal space

!!! summary "Learning goals"

    After this lecture you will be able to:

    - Define the reciprocal space, and explain its relevance
    - Construct a reciprocal lattice from a given real space lattice
    - Compute the intensity of X-ray diffraction of a given crystal
    - Interpret X-ray powder diffraction data

In the last lecture, we introduced crystallographic terminology in order to be able to discuss and analyze crystal structures.
In this lecture, we will 1) study how real-space lattices give rise to lattices in reciprocal space (with the goal of understanding dispersion relations) and 2) consider how to probe crystal structures using X-ray diffraction experiments.

## Recap: the reciprocal lattice in one dimension

In [lecture 7](7_tight_binding.md) we discussed the reciprocal space of a simple 1D lattice with lattice points $x_n = na$, where $n$ is an integer and $a$ is the spacing between the lattice points. To obtain the dispersion relation, we considered waves of the form
$$
e^{ikx_n} = e^{ikna}, \quad n \in \mathbb{Z},
$$
We then observed that waves with wavevectors $k$ and $k+G$, where $G=2\pi m/a$ with integer $m$, are exactly the same: 
$$
e^{i(k+G)na} = e^{ikna+im2\pi n} =  e^{ikna},
$$
because
$$
e^{iGx_n} = e^{i2\pi mn} =  1.
$$
The set of points $G=2\pi m/a$ forms the **reciprocal lattice** in $k$-space.

Let us now generalize this idea to describe reciprocal lattices in three dimensions.

## The reciprocal lattice in three dimensions

We start from a lattice in real space:
$$
\mathbf{R}=n_1\mathbf{a}_1+n_2\mathbf{a}_2+n_3\mathbf{a}_3, \quad \{n_1, n_2, n_3\} \in \mathbb{Z},
$$
where the $\{\mathbf{a}_i\}$ are primitive lattice vectors. The reciprocal lattice is:
$$
\mathbf{G}=m_1\mathbf{b}_1+m_2\mathbf{b}_2+m_3\mathbf{b_3}, \quad \{m_1, m_2, m_3\} \in \mathbb{Z},
$$
where $\{\mathbf{b}_i\}$ are the primitive lattice vectors of the reciprocal lattice.

We determine these vectors $\{\mathbf{b}_i\}$ by requiring that waves with wavevectors that differ by a reciprocal lattice vector $\mathbf{G}$ are indistinguishable:
$$
e^{i\mathbf{k}\cdot\mathbf{R}} = e^{i(\mathbf{k} + \mathbf{G})\cdot\mathbf{R}},
$$
for any lattice vector $\mathbf{R}$. Substituting the definitions of $\mathbf{R}$ and $\mathbf{G}$, we get

$$
e^{i\mathbf{G}\cdot\mathbf{R}} = e^{i\sum_{\{i,j\}=1}^{3} n_i m_j \mathbf{a}_i \mathbf{b}_j}=1,
$$
which holds only if

$$
\mathbf{a_i}\cdot\mathbf{b_j}=2\pi\delta_{ij}.
$$

Indeed, after expanding the dot products in the exponent, we get
$$
\mathrm{e}^{i\mathbf{G}\cdot\mathbf{R}} = \mathrm{e}^{2\pi i(n_1 m_1 + n_2 m_2 + n_3 m_3)}.
$$
Because $n_i$ and $m_j$ are both integers, the exponent evaluates to 1.

The relation $\mathbf{a_i}\cdot\mathbf{b_j}=2\pi\delta_{ij}$ implies that if we write the $\{\mathbf{a_i}\}$ as rows of a matrix, the reciprocal lattice vectors are $2\pi$ times the columns of the inverse of that matrix.

### Example: the reciprocal lattice of a 2D triangular lattice

To gain extra intuition for the properties of the reciprocal lattice, we study an example.

In the previous lecture we studied the triangular lattice shown in the figure below.
The left panel shows the real-space lattice with primitive lattice vectors $\mathbf{a}_1 = a \mathbf{\hat{x}}$ and $\mathbf{a}_2 = a/2\mathbf{\hat{x}} + \sqrt{3}a/2 \mathbf{\hat{y}}$. The right panel shows the corresponding reciprocal lattice and its primitive lattice vectors $\mathbf{b}_1$ and $\mathbf{b}_2$.

```python
# Define primitive lattice vectors
a1 = np.array([1,0])
a2 = np.array([0.5,sqrt(3)/2])
# Compute reciprocal lattice vectors
b1,b2 = np.linalg.inv(np.array([a1,a2]).T) @ np.eye(2)*2*pi

fig = make_subplots(rows=1, cols=2,shared_yaxes=True,subplot_titles=('Real space', 'Reciprocal space'))

# Generates the lattice given the lattice vectors
def lattice_generation(a1,a2,N):
    grid = np.arange(-N//2,N//2,1)
    xGrid, yGrid = np.meshgrid(grid,grid)
    return np.reshape(np.kron(xGrid.flatten(),a1),(-1,2))+np.reshape(np.kron(yGrid.flatten(),a2),(-1,2))


def subplot(a1,a2,col):
    N = 6
    lat_points = lattice_generation(a1,a2,N)
    line_a1 = np.transpose([[0,0],a1])
    line_a2 = np.transpose([[0,0],a2])
    dotLine_a1 = np.transpose([a1,a1+a2])
    dotLine_a2 = np.transpose([a2,a1+a2])


    fig.add_trace(
        go.Scatter(visible=False, x=line_a1[0],y=line_a1[1],mode='lines',line_color='red'
        ), row = 1, col = col
    )
    fig.add_trace(
        go.Scatter(visible=False, x=line_a2[0],y=line_a2[1], mode='lines',line_color='red'
        ), row = 1, col = col
    )
    fig.add_trace(
        go.Scatter(visible=False, x=dotLine_a1[0],y=dotLine_a1[1],mode='lines',line_color='red',line_dash='dot'
        ), row = 1, col = col
    )
    fig.add_trace(
        go.Scatter(visible=False, x=dotLine_a2[0],y=dotLine_a2[1],mode='lines',line_color='red',line_dash='dot'
        ), row = 1, col = col
    )

    fig.add_trace(
        go.Scatter(visible=False, x=lat_points.T[0],y=lat_points.T[1],mode='markers',marker=dict(
            color='Black',
            size = 10
            )
        ), row = 1, col = col
    )

# Generate subplots to be used by the slider
N_values = 10
for i in np.linspace(2.5,3.5,N_values):
    subplot(a1*i,a2*i,1)
    subplot(b1/i,b2/i,2)

# Define the default subplot 
active = 4
for i in range(10):   
    fig.data[active*10+i].visible = True

steps = []
for i in range(N_values):
    step = dict(
        label = 'Lattice Constant',
        method="restyle",
        args=["visible", [False] * len(fig.data)],
    )
    for j in range(10):
        step["args"][1][i*10+j] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    tickcolor = 'White',
    font_color = 'White',
    currentvalue_font_color = 'Black',
    active = active,
    name = 'Lattice Constant',
    steps = steps
)]

# Annotating lattice vectors
def text_dict(text,pos,ref=('x1','y1')):
    dic = {
            'x' : pos[0],
            'y' : pos[1],
            'xref' : ref[0],
            'ayref' : ref[1],
            'text' : text,
            'font' : {
                'size' : 15,
                'color' : 'black'
            },
            'showarrow' : False
    }
    return dic
annotations = [a.to_plotly_json() for a in fig["layout"]["annotations"]]
annotations.append(text_dict(r'$\mathbf{a}_1$',(1.5,-0.5)))
annotations.append(text_dict(r'$\mathbf{a}_2$',(0,1.3)))
annotations.append(text_dict(r'$\mathbf{b}_1$',(0.8,-1),('x2','y2')))
annotations.append(text_dict(r'$\mathbf{b}_2$',(-0.6,1.2),('x2','y2')))
fig["layout"]["annotations"] = annotations


plot_range = 5
fig.update_layout(
    sliders=sliders,
    showlegend = False,
    plot_bgcolor = 'rgb(254, 254, 254)',
    width = 800,
    height = 400,
    xaxis = dict(
        range=[-plot_range,plot_range],
        visible = False,
        showgrid = False,
    ),
    yaxis = dict(
      range = [-plot_range,plot_range],
      visible = False,
      showgrid = False,
    )
)
fig.update_xaxes(range=[-plot_range, plot_range], row=1, col=2, visible=False)
fig.update_yaxes(row=1, col=2, scaleanchor="x", scaleratio=1)
fig.update_yaxes(row=1, col=1, scaleanchor="x", scaleratio=1)
fig
```

To find the reciprocal lattice vectors $\{\mathbf{b_j}\}$, we use the relation

$$
\mathbf{a_i}\cdot\mathbf{b_j}=2\pi\delta_{ij},
$$

which gives us the following equations:

$$
\mathbf{a}_1\cdot\mathbf{b}_2=\mathbf{a}_2\cdot\mathbf{b}_1=0,
$$

and

$$
\mathbf{a}_1\cdot\mathbf{b}_1=\mathbf{a}_2\cdot\mathbf{b}_2=2\pi.
$$

We substitute $\mathbf{a_i}\cdot\mathbf{b_i} = \lvert \mathbf{a_i} \rvert \lvert \mathbf{b_i} \rvert \cos(\theta_i)$ to get:

$$
\lvert \mathbf{a}_1 \rvert \lvert \mathbf{b}_1 \rvert =\frac{2\pi}{\cos(\theta_1)} \:\: \text{and} \:\: \lvert \mathbf{a}_2 \rvert \lvert \mathbf{b}_2 \rvert =\frac{2\pi}{\cos(\theta_2)},
$$

where $\theta_i$ is the angle between the vectors $\mathbf{a}_i$ and $\mathbf{b}_i$.
To find the angles $\theta_1$ and $\theta_2$, we use the orthogonality relations above and the fact that the angle between $\mathbf{a}_1$ and $\mathbf{a}_2$ is $60^\circ$.
From this we conclude that $\theta_1 = \theta_2 = 30^\circ$.
Because $\lvert \mathbf{a}_1 \rvert = \lvert \mathbf{a}_2 \rvert = a$, we find

$$
\lvert \mathbf{b}_1 \rvert = \lvert \mathbf{b}_2 \rvert = \frac{4\pi}{a\sqrt{3}}. 
$$

We find that the lengths of the reciprocal lattice vectors are equal and inversely proportional to the lattice constant $a$.
With $\lvert \mathbf{b}_2 \rvert$ and $\mathbf{a}_1 \perp \mathbf{b}_2$, we find 

$$
\mathbf{b}_2 = \frac{4\pi}{a\sqrt{3}} \mathbf{\hat{y}}.
$$

We follow the same procedure to find $\mathbf{b}_1$:

$$
\mathbf{b}_1 = \frac{4\pi}{a\sqrt{3}} \left(\frac{\sqrt{3}}{2} \mathbf{\hat{x}} - \frac{1}{2}\mathbf{\hat{y}} \right).
$$

??? Question "Is the choice of a set of primitive reciprocal lattice vectors unique? If not, which other ones are possible?"
    As in real space, there are many ways to choose primitive reciprocal lattice vectors that form the same reciprocal lattice. In the example above we could just as well use

    $$
    \mathbf{b}_1 = \frac{4\pi}{a\sqrt{3}} \left(-\frac{\sqrt{3}}{2} \mathbf{\hat{x}} + \frac{1}{2}\mathbf{\hat{y}} \right) \quad \text{and} \quad \mathbf{b}_2 = -\frac{4\pi}{a\sqrt{3}} \mathbf{\hat{y}}.
    $$

    There is however only one choice that satisfies $\mathbf{a_i}\cdot\mathbf{b_j}=2\pi\delta_{ij}$.

### Constructing a 3D reciprocal lattice 

Let us now consider a more involved example of the 3D lattice.
The explicit expression for the reciprocal lattice vectors in terms of their real-space counterparts is:

$$
\mathbf{b}_1=\frac{2\pi(\mathbf{a}_2\times\mathbf{a}_3)}{ \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a_3})}
$$

$$
\mathbf{b}_2=\frac{2\pi(\mathbf{a_3}\times\mathbf{a}_1)}{ \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a_3})}
$$

$$
\mathbf{b_3}=\frac{2\pi(\mathbf{a}_1\times\mathbf{a}_2)}{ \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a_3})}
$$

Note that the denominator $\mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a_3})$ is the volume $V$ of the unit cell spanned by the lattice vectors $\mathbf{a}_1$, $\mathbf{a}_2$ and $\mathbf{a}_3$. 

### The reciprocal lattice as a Fourier transform

One can also think of the reciprocal lattice as a Fourier transform of the real-space lattice. 
For simplicity, we illustrate this for a 1D lattice (the same principles apply for a 3D lattice).
We model the real-space lattice as a density function consisting of delta peaks:

$$
\rho(x)=\sum_{n} \delta(x-na)
$$

We take the Fourier transform of this function to find:

$$
{\mathcal F}_{k}\left[\rho(x)\right]=\int_{-\infty}^\infty \mathrm{d}x\ \mathrm{e}^{ikx} \rho(x)=\sum_{n} \int_{-\infty}^\infty \mathrm{d}x\ \mathrm{e}^{ikx} \delta(x-na)=\sum_{n} \mathrm{e}^{ikna}
$$

This sum is non-zero only if $k=2\pi m/a$.
If we recall the beginning of the lecture, then these points correspond to reciprocal lattice points $G$. 
Therefore, we rewrite this into the form

$$
{\mathcal F}_{k}\left[\rho(x)\right]=\frac{2\pi}{|a|}\sum_{m} \delta\left(k-G\right).
$$

Therefore, we see that the Fourier transform is non-zero only at reciprocal lattice points.
In other words, Fourier transforming a real-space lattice yields a reciprocal lattice! 
The above result generalizes directly to three dimensions:

$$
{\mathcal F}_\mathbf{k}\left[\rho(\mathbf{r})\right]=\int \mathrm{d}\mathbf{r}\ \mathrm{e}^{i\mathbf{k}\cdot\mathbf{r}} \rho(\mathbf{r}) = \sum_\mathbf{G}\delta(\mathbf{k}-\mathbf{G}).
$$

## The importance of the 1st Brillouin zone

We have now seen how the structure of the reciprocal lattice is directly determined by the structure of the real-space lattice. An important reason to study the reciprocal lattice is that we are often interested in understanding the dispersion relation of electronic or vibrational modes in a material. And because waves with wavevectors differing by a reciprocal lattice vector $\mathbf{G}$ are identical, we only need to understand the dispersion in a single primitive unit cell of the reciprocal lattice. But what unit cell to choose? We learned that the choice of a primitive unit cell is not unique. 

A general convention in reciprocal space is to use the Wigner-Seitz cell, which is also called the **1st Brillouin zone**.
Because the Wigner-Seitz cell is primitive, the 1st Brillouin zone (1BZ) contains a set of unique $\mathbf{k}$-vectors.
This means that any wavevector $\mathbf{k'}$ outside the 1st Brillouin zone is related to a wavevector $\mathbf{k}$ inside the first Brillouin Zone by shifting it by a reciprocal lattice vector: $\mathbf{k'} = \mathbf{k}+\mathbf{G}$. 

In the previous lecture we already discussed how to construct Wigner-Seitz cells. However, here is a reminder of such a cell:

![](figures/brillouin_mod.svg)

## Determining crystal structures using diffraction experiments

### The Laue condition

Another reason to understand the reciprocal lattice is that it manifests directly in diffraction experiments. Such experiments are some of our most powerful tools for determining the crystal structure of materials.

In a diffraction experiment, a beam of high-energy waves or particles (e.g. X-rays, neutrons, or electrons) is directed at a material of interest . As a result of interference, the scattered radiation pattern reveals the reciprocal lattice of the crystal. To understand how this works, consider an incoming wave with wavevector $\mathbf{k}$ that scatters off two atoms separated by a lattice vector $\mathbf{R}$ into an outgoing wave with wavevector $\mathbf{k'}$ (see figure). We assume that the scattering is elastic (does not cause an energy loss), such that $|\mathbf{k'}|=|\mathbf{k}|$.

![](figures/scattering.svg)

Observe that the lower ray travels a larger distance than the upper ray. This results in a phase shift $\Delta \phi$ between these rays. 
With a bit of geometry, we find that the extra distance traveled by the lower ray relative to the upper one is 

$$
x_{\mathrm{extra}} = \Delta x_1+\Delta x_2 = \cos(\theta) \lvert R \rvert + \cos(\theta') \lvert R \rvert.
$$

The corresponding phase difference is: 

\begin{align*}
\Delta \phi &= \lvert\mathbf{k} \rvert(\Delta x_1+\Delta x_2)\\
&= \lvert\mathbf{k}\rvert \lvert\mathbf{R}\rvert(\cos(\theta)+\cos(\theta'))\\
&= \mathbf{k'}\cdot \mathbf{R} - \mathbf{k}\cdot \mathbf{R} = (\mathbf{k'} - \mathbf{k}) \cdot \mathbf{R}.
\end{align*}

However, that is only a phase difference between waves scattered off of two atoms.
To find the outgoing wave's amplitude, we must sum over the scattered waves from each and every atom in the lattice:

$$
A\propto\sum_\mathbf{R}\mathrm{e}^{i\left(\Delta \phi-\omega t\right)} = \sum_\mathbf{R}\mathrm{e}^{i\left((\mathbf{k'}-\mathbf{k})\cdot\mathbf{R}-\omega t\right)}.
$$

This sum is non-zero if and only if the scattered waves interfere constructively, i.e., the phase difference equals $2\pi n$, where $n$ is an integer. Furthermore, we know that real and reciprocal lattice vectors are related by $\mathbf{G} \cdot \mathbf{R} = 2 \pi n$.
Therefore, we conclude that the difference between the incoming and outgoing waves must be:

$$
\mathbf{k'}-\mathbf{k}=\mathbf{G}.
$$

to get constructive interference. In other words, we can only get constructive interference at very specific angles, as determined by the structure of the reciprocal lattice. This requirement is known as the _Laue condition_.
As a result, the interference pattern produced in diffraction experiments is a direct measurement of the reciprocal lattice!

### The structure factor

Above we assumed that the unit cell contains only a single atom.
What if the basis contains more atoms though?
In the figure below we show a simple lattice which contains multiple atoms in the unit cell.
Note, the unit cell does not have to be primitive!

![](figures/laue_mod.svg)

Let $\mathbf{R}$ be the lattice and let $\mathbf{R}+\mathbf{r}_j$ be the location of the atoms in the unit cell.
The distance $\mathbf{r}_j$ is taken with respect to lattice point from which we construct the unit cell.
Similar to before, we calculate the amplitude of the scattered wave.
However, now there are multiple atoms in the unit cell and each of these atoms acquires a phase shift of its own.
In order to keep track of the atoms, we define $\mathbf{r}_j$ to be the location of atom $j$ in the unit cell.
The distance $\mathbf{r}_j$ is defined with respect to the lattice point from which we construct the unit cell.
In order to calculate the amplitude of the scattered wave, we must sum not only over all the lattice points but also over the atoms in a single unit cell:

\begin{align*}
A &\propto \sum_\mathbf{R} \sum_j f_j \mathrm{e}^{i\left(\mathbf{G}\cdot(\mathbf{R}+\mathbf{r}_j)-\omega t\right)}\\
&= \sum_\mathbf{R}\mathrm{e}^{i\left(\mathbf{G}\cdot\mathbf{R}-\omega t\right)}\sum_j f_j\ \mathrm{e}^{i\mathbf{G}\cdot\mathbf{r}_j}
\end{align*}

where $f_j$ is the scattering amplitude off of a single atom, and it is called the *form factor*.
The form factor mainly depends on the chemical element, nature of the scattered wave, and finer details like the electrical environment of the atom.
The first part of the equation above is the familiar Laue condition, and it requires that the scattered wave satisfies the Laue condition. 
The second part gives the amplitude of the scattered wave, and it is called the **structure factor**:

$$
S(\mathbf{G})=\sum_j f_j\ \mathrm{e}^{i\mathbf{G}\cdot\mathbf{r}_j}.
$$

In diffraction experiments, the intensity of the scattered wave is $I \propto A^2$
Therefore, the intensity of a scattered wave depends on the structure factor $I \propto S(\mathbf{G})^2$.
Because the structure factor depends on the form factors and the positions of the basis atoms, by studying the visibility of different diffraction peaks we may learn the locations of atoms within the unit cell.

### The Laue condition and structure factor for non-primitive unit cells

Laue conditions allow scattering as long as the scattering wave vector is a reciprocal lattice vector.
However if we consider a non-primitive unit cell of the direct lattice, the reciprocal lattice contains more lattice points, seemingly leading to additional interference peaks.
Computing the structure factor allows us to resolve this apparent contradiction.

??? Question "Calculate the structure factor in which there is a single atom in the unit cell, which is located at the lattice point. Do any diffraction peaks dissapear?"
    $\mathbf{r}_1=(0,0,0)\rightarrow S=f_1$. 
    In this case, each reciprocal lattice point gives one interference peak, none of which are absent.


As a demonstration of how it happens, let us compute the structure factor of the FCC lattice using the conventional unit cell in the real space.

![](figures/fcc.svg)

The basis of the conventional FCC unit cell contains four identical atoms.
With respect to the reference lattice point, these attoms are located at

\begin{align*}
\mathbf{r}_1&=(0,0,0)\\
\mathbf{r}_2&=\frac{1}{2}(\mathbf{a}_1+\mathbf{a}_2)\\
\mathbf{r}_3&=\frac{1}{2}(\mathbf{a}_2+\mathbf{a}_3)\\
\mathbf{r}_4&=\frac{1}{2}(\mathbf{a}_3+\mathbf{a}_1),
\end{align*}

with $f_1=f_2=f_3=f_4\equiv f$. Let the reciprocal lattice be described by $\mathbf{G}=h\mathbf{b}_1+k\mathbf{b}_2+l\mathbf{b}_3$, where $h$, $k$ and $l$ are integers. Using the basis described above and the reciprocal lattice, we calculate the structure factor:

\begin{align*}
S&=f\left(\mathrm{e}^{i\mathbf{G}\cdot\mathbf{r}_1}+\mathrm{e}^{i\mathbf{G}\cdot\mathbf{r}_2}+\mathrm{e}^{i\mathbf{G}\cdot\mathbf{r}_3}+\mathrm{e}^{i\mathbf{G}\cdot\mathbf{r}_4}\right)\\
&=f\left(1+\mathrm{e}^{i(h\mathbf{b}_1\cdot\mathbf{a}_1+k\mathbf{b}_2\cdot\mathbf{a}_2)/2}+\mathrm{e}^{i(k\mathbf{b}_2\cdot\mathbf{a}_2+l\mathbf{b}_3\cdot\mathbf{a}_3)/2}+\mathrm{e}^{i(h\mathbf{b}_1\cdot\mathbf{a}_1+l\mathbf{b}_3\cdot\mathbf{a}_3)/2}\right)\\
&=f\left(1+\mathrm{e}^{i\pi(h+k)}+\mathrm{e}^{i\pi(k+l)}+\mathrm{e}^{i\pi(h+l)}\right).
\end{align*}

Because $h$, $k$ and $l$ are integers, all exponents are either $+1$ or $-1$, and the interference is only present if

$$
S = 
\begin{cases}
    4f, \: \mathrm{if} \: h, \: k, \: \mathrm{and} \: l \: \mathrm{are \: all \: even \: or \: odd,}\\
    0, \: \mathrm{in \: all \: other \: cases}.
\end{cases}
$$

We now see that the reciprocal lattice points with nonzero amplitude exactly form the reciprocal lattice of the FCC lattice.

### Powder diffraction

The easiest way to do diffraction measurements is to take a crystal, shoot an X-ray beam through it and measure the direction of outgoing waves. 
However growing a single crystal may be hard because many materials are polycrystalline

A simple alternative is to perform **powder diffraction**.
Crushing the crystal into a powder results in many small crystallites that are oriented in random directions.
This improves the chances of fulfilling the Laue condition for a fixed direction incoming beam. 
The experiment is illustrated in the figure below.
The result is that the diffracted beam exits the sample via concentric circles at discrete **deflection angles** $2 \theta$.

```python
def add_patch(ax, patches, *args,**kwargs):
    for i in patches:
        ax.add_patch(i,*args,**kwargs)
        
def circle(radius,xy=(0,0),**kwargs):
    return patches.Circle(xy,radius=radius, fill=False, edgecolor='r', lw = 2, **kwargs)

fig, ax = plt.subplots(figsize=(7,7))

transform=mtransforms.Affine2D().skew_deg(0,-25) + ax.transData
# Create the screen
rect = patches.Rectangle((-0.5,-0.5),1,1, edgecolor = 'k', lw = 2, facecolor = np.array([217, 217, 217])/255,transform = transform)
circle_list = [circle(i,transform=transform) for i in np.array([0.001,0.02,0.08,0.15,0.2,0.22,0.25])*2]
add_patch(ax,[rect]+circle_list)

# Add sample
sample_pos = np.array([-0.6,-0.6])
ax.add_patch(patches.Circle(sample_pos,radius=0.1,color='k',zorder=10))
plt.annotate('Powder Sample',sample_pos+[-0.1,-0.2],fontsize=14)
#Reference line
ax.plot([sample_pos[0],0],[sample_pos[1],0],color='k',ls='--')

#X-Ray Beam
d_xray = sample_pos-np.array([-1+0.05,-1+0.05])
ax.add_patch(patches.Arrow(-1,-1, *d_xray, width=0.05, color='r'))
plt.annotate('X-Ray Beam',(-1,-0.85),fontsize=14,rotation = 45)

# Diffracted Beams
ax.add_patch(patches.Arrow(*sample_pos, 0.1, 0.8, width=0.05, color='r'))
ax.add_patch(patches.Arrow(*sample_pos, 0.8, 0.285, width=0.05, color='r'))

#Angle Arcs
ellipse_radius = 0.3
ax.add_patch(patches.Arc(sample_pos, ellipse_radius, ellipse_radius, angle=80, theta1=325, theta2=0))
plt.annotate('$ 2\\theta $',(-0.56,-0.44),fontsize=14)


plt.xlim([-1,0.5])
plt.ylim([-1,0.5])
plt.axis('off');
```

To deduce the values of $\theta$ of a specific crystal, let us put the Laue condition into a more practical form.
We first take the modulus squared of both sides:

\begin{align*}
\left|\mathbf{G}\right|^2 &= \left|\mathbf{k'}-\mathbf{k} \right|^2 \\
G^2 &=  2k^2-2\mathbf{k'} \cdot \mathbf{k},
\end{align*}

where we used $|\mathbf{k'}| = |\mathbf{k}|$.
We then substitute the Laue condition $\mathbf{k'} = \mathbf{k}+\mathbf{G}$:

\begin{align*}
\lvert \mathbf{G} \rvert ^2 &= 2k^2-2 \left(\mathbf{k}+\mathbf{G}\right) \cdot \mathbf{k} \\
&= -2 \mathbf{G} \cdot \mathbf{k}.
\end{align*}

Using $\mathbf{k} \cdot \mathbf{G} = \lvert \mathbf{k} \rvert \lvert \mathbf{G} \rvert \cos(\phi)$,  we obtain

$$
\left| \mathbf{G} \right| = -2 \lvert \mathbf{k} \rvert \cos (\phi).
$$

Note, $\phi$ is the angle between the vector $\mathbf{k}$ and $\mathbf{G}$, which is not the same as the angle between the incoming and scattered waves $\theta$.
We are nearly there, but we are left with finding out the relation between $\phi$ and $\theta$.

Recall the concept of Miller planes. 
These are sets of planes identified by their Miller indices $(h,k,l)$ which intersect the lattice vectors at $\mathbf{a}_1 / h$, $\mathbf{a}_2 / k$ and $\mathbf{a}_3 / l$.
It turns out that Miller planes are normal to the reciprocal lattice vector $\mathbf{G} = h \mathbf{b}_1 + k \mathbf{b}_2 + l \mathbf{b}_3$ and the distance between subsequent Miller planes is $d_{hkl} = 2 \pi/\lvert \mathbf{G} \rvert$ (you will derive this in [today's exercise](10_xray.md#exercise-2-miller-planes-and-reciprocal-lattice-vectors)).
Substituting the expression for $\lvert \mathbf{G} \rvert$ into the expression for the distance between Miller planes we get:

$$ 
d_{hkl} \cos (\phi) = -\frac{\pi}{\lvert \mathbf{k} \rvert}.
$$ 
 
We know that $\lvert \mathbf{k} \rvert$ is related to the wavelength by $\lvert \mathbf{k} \rvert = 2\pi/\lambda$.
Therefore, we can write the equation above as

$$ 
2 d_{hkl} \cos (\phi) = -\lambda.
$$ 

Lastly, we express the equation in terms of the deflection angle through the relation $\phi = \theta + \pi/2.$
With this, one can finally derive **Bragg's Law**:

$$ 
\lambda = 2 d_{hkl} \sin(\theta) 
$$

Bragg's law allows us to obtain atomic distances in the crystal $d_{hkl}$ through powder diffraction experiments!

## Summary

* We described how to construct a reciprocal lattice from a real-space lattice.
* Points in reciprocal space that differ by a reciprocal lattice vector are equivalent. 
* Diffraction experiments reveal information about crystal structure.
* Laue condition: difference between wavevectors of incoming and diffracted waves matches a reciprocal lattice vector, necessary for constructive interference.
* Structure factor: describes the contribution of the atoms in a unit cell to the diffraction pattern.
* Powder diffraction and relating its experimental results to the crystal structure via Bragg's law.

## Exercises

### Warm up exercises*

1. Study the 1D phonon dispersion relation that was plotted in the Tight-binding model lecture. Identify the reciprocal lattice points and the first Brillouin zone. Confirm that the first Brillouin zone is the Wigner Seitz unit cell of the reciprocal lattice. 
2. Use the [scalar triple product](https://en.wikipedia.org/wiki/Triple_product#Scalar_triple_product) and the definitions of the 3D reciprocal lattice vectors given in the lecture to calculate $\mathbf{a}_1 \cdot \mathbf{b}_1$ and $\mathbf{a}_2 \cdot \mathbf{b}_1$. Discuss if the result is as expected.
3. Why is the amplitude of a scattered wave zero if $\mathbf{k'}-\mathbf{k} \neq \mathbf{G}$?
4. Suppose we have a unit cell with a single atom in it. Can any intensity peaks dissapear as a result of the structure factor? 
5. Can increasing the unit cell in real space introduce new diffraction peaks due to reciprocal lattice having more points?

### Exercise 1*: The reciprocal lattice of the bcc and fcc lattices

In this lecture, we studied how to construct the reciprocal lattice from a real-space lattice. We will now zoom in the properties of the fcc and bcc lattices, which are two of the most common lattices encountered in crystal structures. We analyze the reciprocal lattices and the shape of the first Brillouin zone. This helps us understanding e.g. the periodicity of 3D band structures and the Fermi surface database shown in the Attic.

We consider a bcc lattice of which the conventional unit cell has a side length $a$ and volume $V=a^3$.

1. Write down the primitive lattice vectors of the [BCC lattice](https://solidstate.quantumtinkerer.tudelft.nl/9_crystal_structure/#body-centered-cubic-lattice). Calculate the volume of the primitive unit cell. Is the volume the expected fraction of $V$?

2. Calculate the reciprocal lattice vectors associated with the primitive lattice vectors you found in the previous subquestion.

3. Sketch the reciprocal lattice. Which type of lattice is it? What is the volume of its conventional unit cell?

4. Describe the shape of the 1st Brillouin zone. How many sides does it have? (Note that the Brillouin zones are sketched in the Fermi surface periodic table in the Attic). Calculate the volume of the 1st Brillouin zone and check if it is the expected fraction of the volume you found in the previous subquestion.

5. Based on the insight gained in this question, argue what lattice is the reciprocal lattice of the _fcc_ lattice.

### Exercise 2: Miller planes and reciprocal lattice vectors

Miller indices are central to the description of the various planes in crystals. In this question we will analyze the Miller indices and their associated reciprocal lattice vectors, and show that the distance between Miller planes follows from the length of these vectors. We also highlight the convenience of using the conventional unit cell for describing cubic crystal structures.

Consider a family of Miller planes $(hkl)$ in a crystal.

1. Prove that the reciprocal lattice vector $\mathbf{G} = h \mathbf{b}_1 + k \mathbf{b}_2 + l \mathbf{b}_3$ is perpendicular to the Miller plane $(hkl)$.

    ??? hint
        Choose two vectors that lie within the Miller plane and are not parallel to each other.

2. Show that the distance between two adjacent Miller planes $(hkl)$ is $d = 2\pi/|\mathbf{G}_\textrm{min}|$, where $\mathbf{G}_\textrm{min}$ is the shortest reciprocal lattice vector perpendicular to these Miller planes.

3. In exercise 1, you derived the reciprocal lattice vectors of the BCC lattice from a set of  primitive lattice vectors. Use these vectors to find the family of Miller planes that has the highest density of lattice points $\rho$. Use that $\rho = d/V$, where $V$ is the volume of the primitive unit cell and $d$ is the distance between adjacent planes derived in the previous subquestion. Formulate the Miller plane indices with respect to the primitive lattice vectors.

4. Make a sketch of the BCC structure and identify a Miller plane with the highest density of lattice points. Hint: it may help to make a sketch of the projections of the real-space lattice vectors $\{\mathbf{a_i}\}$ onto the $xy$ plane to identify which plane the Miller indices correspond to.

5. For cubic crystal structures, the interpretation of Miller indices strongly simplifies by using the lattice vectors of the conventional instead of a primitive unit cell. However, when doing so not all reciprocal lattice vectors correspond to a family of lattice planes. The reason is that such a family, by definition, must contain all lattice points.

    Consider the reciprocal lattice vector $\mathbf{G} = h\mathbf{b_1} +k\mathbf{b_2} + l\mathbf{b_3}$, constructed from the conventional unit cell, such that $\mathbf{b_1}=2\pi/a\mathbf{\hat{x}}$, $\mathbf{b_2}=2\pi/a\mathbf{\hat{y}}$, $\mathbf{b_3}=2\pi/a\mathbf{\hat{z}}$. Do the indices $(hkl) = (100)$ correspond to a family of lattice planes? And $(200)$?. Discuss why (not).

6. As in the previous subquestion, here we still consider the reciprocal lattice constructed from the conventional unit cell. It turns out that, to understand which sets of $(hkl)$ indices can describe families of lattice planes, we can use the structure factor. Calculate (or recall from a previous question) the structure factor for the BCC lattice and discuss which sets of $(hkl)$ indices describe Miller planes. From this, identify the Miller planes with the highest density of lattice points and check if you got the same result as in subquestion 4.

### Exercise 3: X-ray scattering in 2D

*(adapted from ex 14.1 and ex 14.3 of "The Oxford Solid State Basics" by S.Simon)*

Using x-ray scattering, we can infer information on the crystal structure of a material. Here we visualize the geometry of this procedure by analyzing an elementary 2D crystal structure.

Consider a two-dimensional crystal with a rectangular lattice and primitive lattice vectors $\mathbf{a}_1 = d_1\mathbf{\hat{x}}$ and $\mathbf{a}_2 = d_2\mathbf{\hat{y}}$, where $d_1=0.47$ nm and $d_2=0.34$ nm. We conduct an X-ray scattering experiment using monochromatic X-rays with wavelength $\lambda = 0.166$ nm. The wavevectors of the incident and reflected X-ray beams are $\mathbf{k}$ and $\mathbf{k'}$ respectively.

1. Calculate the reciprocal lattice vectors and sketch both the real- and the reciprocal lattice of this crystal.
2. Consider an X-ray diffraction experiment performed on this crystal using monochromatic X-rays with wavelength $\lambda = 0.166$ nm. By assuming elastic scattering, find the magnitude $k$ of the wavevectors of the incident and reflected X-ray beams.
3. In the sketch of the real-space lattice of subquestion 1, indicate a (210) Miller plane. Indicatet the associated reciprocal lattice vector $\mathbf{G}$ in the sketch of the reciprocal lattice. Also sketch the "scattering triangle" formed by the vectors $\mathbf{k}$, $\mathbf{k'}$, and $\mathbf{G}$ corresponding to diffraction from (210) planes.
4. Sketch the first 5 peaks in an x-ray powder diffraction spectrum of this crystal as a function of $\sin 2\theta$, where $\theta$ is the deflection angle. Label the peaks according the Miller indices. Make sure you have the correct order of the peaks. Are there missing peaks because of the structure factor?

    ??? Hint
        Use the result of exercise 2 to express the distance between Miller planes in terms of the length of the reciprocal lattice vector $d_{hkl} = 2\pi/|\mathbf{G_{hkl}}|$

### Exercise 4: Analyzing a 3D power diffraction spectrum

In this question, we analyze the diffraction pattern we expect for an x-ray experiment on a 3D material with a BCC crystal structure.

1. Using a conventional unit cell plus a basis to construct the BCC crystal structure, calculate the structure factor $\mathbf{S}$. (assume all the atoms to be the same).
2. Which diffraction peaks are missing because of the structure factor? Discuss why they are missing in relation to the crystal structure and the conventional unit cell.
3. How does this structure factor change if the atom in the center of the conventional unit cell has a different form factor from the atoms at the corners?
4. A student carried out an X-ray powder diffraction experiment on chromium (Cr) which is known to have a BCC structure. The measured spectrum is shown given below. Furthermore, the student assigned Miller indices to the peaks. Were these indices assigned correctly? Fix any mistakes and explain your reasoning.
![](figures/cr_xray_exercise.svg)
5. Calculate the lattice constant $a$ of the conventional chromium bcc unit cell. Use that the X-ray diffraction experiment was carried out using Cu K-$\alpha$ (wavelength $\lambda = 1.5406$ Å) radiation.
