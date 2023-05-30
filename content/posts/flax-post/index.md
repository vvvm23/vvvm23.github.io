---
title: "Flax Post"
date: 2023-05-22T21:54:11+01:00
draft: true
---

In an earlier blog post, I introduced JAX - a framework for high performance numerical computing / machine learning - in a somewhat atypical manner. I didn't actually create a single training loop, and only a couple patterns that looked vaguely machine learning-like. 

This was deliberate as I felt that JAX - although designed for machine learning research - is much more general-purpose than that. The steps for using it are to define what you want to happen, wrap it in a `jax.jit` call, let JAX trace out your function into some intermediate graph representation, and then pass to XLA to compile and optimise it. The result is a single, heavily-optimised, binary blob, ready and waiting for you to throw data at. This paradigm is a natural fit for a lot of machine learning applications, but also other scientific computing tasks. So I felt it didn't make sense to get too ML specific there. Also, it is ground that has been explored and taught a lot before - I wanted to do a different take on introductory JAX.

I also mentioned in the previous post that it *is* possible to develop a full machine learning training loop - models, optimisers and all - in pure JAX alone. This is self-evident as it is very general purpose. It can be a good exercise to do so, but not a strategy I like the employ. Therefore, in this blog post I want to introduce two higher level libraries built on top of JAX, that do a lot of the heavy lifting for us when writing our machine learning applications. These are **Flax** and **Optax**.

To summarise the two libraries:
- **JAX** - provides a high-level neural network API that lets the developer reason about the model in terms of components, like in PyTorch, rather than with JAX functions that take parameters as inputs.
- **Optax** - a library containing an array of model training utilities, such as optimisers, loss functions, learning rate schedulers, and more! Very batteries-included.

At the end of this post, we will have implemented and trained a very simple **class-conditioned image generation model** called a **variational autoencoder** (VAE) to generate MNIST digits.

> I am not aware of other libraries that do the same thing as Optax, but there are a lot of neural network APIs built on top of JAX, such as [Equinox](https://github.com/patrick-kidger/equinox/) by [Patrik Kidger](https://kidger.site/), and [Haiku](https://github.com/deepmind/dm-haiku) developed at Deepmind. I simply picked Flax here in order to have perfect interoperability with Huggingface models during the [Huggingface JAX Diffusers sprint](https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint).

## Neural Network API with Flax
- High level pattern of doing a training loop in JAX X
- Which part does Flax address? X
- The ways of defining a module. X
    - regular and compact representation X
- What does that give us?
    - easy parameter init as a PyTree X
    - easy way to call the model (whilst remaining stateless) X
    - easier to reason about for the developer (class based) X
- Model itself is essentially a hollow shell: loosely associating parameters with some ops. X
    - Essentially a helper object. We can imagine a function $f_\Theta(x)$. X
    - In PyTorch, a module is $f_\Theta$ that we can apply $x$ to. X
    - In Flax, a module is literally $f$, which we apply both $\Theta$ and $x$ to. X
    - It makes it quite easy to swap out params. X
    - Show that we can pass any pytree with the correct structure. X
- params as a PyTree makes it interoperate perfectly with not only JAX, but other libraries built on top of JAX. X
    - This modularity is quite common in the JAX ecosystem, we aren't locked into something (typically) if we pick one library. X
- The final result is the same. We get params and ops that is passed to a function. This gets traced and compiled as before. We get an optimised compiled function. X
- There is a way to bind parameters to a model, and yield an interactive model like $f_\Theta$. However, can't train with this, it is a static model.
    - Maybe move to later section... We could even use it in the sample step!
- Flax comes with a bunch of other layers inbuilt. I won't enumerate them all, but all the usual culprits are there. X
    - Keep in mind though, some default initialisers are different which could be an issue if you are porting models from other frameworks to Flax.
- There is a lot more to Flax than this, but enough for now. Main takeaway are:
    - During dev time, Flax helps the developer reason about neural networks in an object-orientated way whilst remaining functional during runtime. X
    - During runtime, `flax.linen` is a helper for creating stateless shells that build PyTrees of parameters and loosely associate said parameters with JAX operations. X
    - Statelessness is important to allow Flax to interoperate with JAX and other libraries built on JAX, but also kinda neat. X

The high level structure of a training loop in pure JAX, looks something like
this:
```python
dataset = ...   # initialise training dataset that we can iterate over
params = ...    # initialise trainable parameters of our model
epochs = ...

def model_forward(params, batch):
    ...         # perform a forward pass of our model on `batch` using `params`
    return outputs

def loss_fn(params, batch):
    model_output = model_forward(params, batch)
    loss = ...  # compute a loss based on `batch` and `model_output`
    return loss

@jax.jit
def train_step(params, batch):
    loss, grad = jax.value_and_grad(loss_fn)(params, batch)
    grads = ...  # transform `grads` (clipping, multiply by learning rate, etc.)
    params = ... # update `params` using `grads` (such as via SGD)
    return params, loss

for _ in range(epochs):
    for batch in dataset:
        params, loss = train_step(params, batch)
        ...         # report metrics and the like
```
We first define our model in a functional manner: simply being a function that
takes in the model parameters and a batch, and returns the output of the model.
Similarly, we define the loss function that also takes the parameters and a
batch, and returns the loss. Our final function is the actual train step itself
which we wrap in a `jax.jit` call – giving XLA maximum context to work with.
This first computes the gradient of the loss function using the function
transform `jax.value_and_grad`, manipulates the returned gradients (perhaps
scaling by a learning rate), and updates the parameters. We return the new
parameters, and use them on the next call to `train_step`. This can be called in
a loop, fetching new data from the dataset each time.

Most (but not all) machine learning programs follow a pattern such as the one
above. In other frameworks such as PyTorch though, typically we can package
together the model forward pass and the management of the parameters into a
stateful class representing our model – simplifying the training loop. It would
be nice if we could imitate this behaviour in stateless JAX to allow the
developer to reason about models in a class-based way. This is what Flax's 
neural network API – `flax.linen` – aims to achieve.

> Whether or not writing models in a purely stateless, functional way is better
than a stateful, class-based way, is not the topic of this blog post. In my
opinion both have merits. Regardless, during execution the result is the same
whether we use Flax or not: a stateless, heavily-optimised, binary blob that we
throw data at. It's all JAX after all.

There are two main ways to define a module in Flax: one is more PyTorch like and
the other is a more compact representation:
```python
import flax.linen as nn
from typing import Callable
class Model(nn.Module):
    dim: int
    activation_fn: Callable = nn.relu

    def setup(self):
        self.layer = nn.Dense(self.dim)

    def __call__(self, x):
        x = self.layer(x)
        return self.activation_fn(x)

class ModelCompact(nn.Module):
    dim: int
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        return self.activation_fn(x)     
```

Both approaches can be useful. If we have some complex initialisation logic, the
former may be more appropriate. If the module is relatively simple, we can make
use of the `nn.compact` representation to automatically define the module by the
forward pass alone.

Like in other frameworks, we can nest modules within each other to implement
complex model behaviour. Like we've already seen, `flax.linen` provides some
prebaked modules like `nn.Dense` (same as PyTorch's `nn.Linear`). I won't
enumerate them all, but the usual candidates are all there like convolutions,
embeddings, and more.

However, as is, we cannot call this model, even if we were to initialise the
class itself. There simply aren't any parameters yet to use, and our module is
never a container for parameters (otherwise it would be stateful!). **An
instance of Flax module is simply a hollow shell, that loosely associates
operations with parameters and inputs** that can be passed in later.

To see what I mean, let's initialise some parameters for our model:
```python
key = jax.random.PRNGKey(0xffff)
key, model_key = jax.random.split(key)

model = Model(dim=4)
params = model.init(model_key, jnp.zeros((1, 8)))
params
===
Out: 
FrozenDict({
    params: {
        layer: {
            kernel: Array([[-0.05412389, -0.28172645, -0.07438638,  0.5238516 ],
                   [-0.13562573, -0.17592733,  0.45305118, -0.0650041 ],
                   [ 0.25177842,  0.13981569, -0.41496065, -0.15681015],
                   [ 0.13783392, -0.6254694 , -0.09966562, -0.04283331],
                   [ 0.48194656,  0.07596914,  0.0429794 , -0.2127948 ],
                   [-0.6694777 ,  0.15849823, -0.4057232 ,  0.26767966],
                   [ 0.22948688,  0.00706845,  0.0145666 , -0.1280596 ],
                   [ 0.62309605,  0.12575962, -0.05112049, -0.316764  ]],      dtype=float32),
            bias: Array([0., 0., 0., 0.], dtype=float32),
        },
    },
})
```
In the above cell, we first initialised our model class, which returns an
instance of `Model` which we assign to the variable `model`. Like we said, this
does not actually contain any parameters, it is just a hollow shell that we can
pass parameters and inputs to, in order to execute our model. We can see this by
printing the `model` variable itself:

```python
model
===
Out: Model(
    # attributes
    dim = 4
    activation_fn = relu
)
```

We can also try calling the model itself, which will fail even though we have seemingly defined the `__call__` method:
```python
model(jnp.zeros((1, 8)))
===
Out:
/usr/local/lib/python3.10/dist-packages/flax/linen/module.py in __getattr__(self, name)
    935         msg += (f' If "{name}" is defined in \'.setup()\', remember these fields '
    936           'are only accessible from inside \'init\' or \'apply\'.')
--> 937       raise AttributeError(msg)
    938 
    939   def __dir__(self) -> List[str]:

AttributeError: "Model" object has no attribute "layer". If "layer" is defined in '.setup()', remember these fields are only accessible from inside 'init' or 'apply'.
```

To actually initialise the parameters themselves, we pass a random key (see
previous post on JAX psuedo-randomness) and some dummy inputs to the model's
`init` function of the same shape and types as the real inputs we will use
later. In this simple case, this is simply the `x` argument in the original
module definition, but could be multiple arrays or other objects like keys. We
need these input shapes in order to determine the shape of the parameters.

From the `model.init` call, we get a nested `FrozenDict` holding our model
parameters. If you are familiar with PyTorch state dictionaries, the format of
the parameters is very similar: nested dictionaries with meaningful named keys,
with parameter arrays as values. If you've read my previous blog post or read
about JAX before, you will also know that this structure is a PyTree. Not only
does Flax help developers loosely associate parameters and operations, it also
helps generating said parameters based on our model definition.

Once we have these parameters, we can call the model using `model.apply` –
providing the parameters and inputs:
```python
key, x_key = jax.random.split(key)
x = jax.random.normal(x_key, (1, 8))
y = model.apply(params, x)
y
===
Out: Array([[0.9296505 , 0.25998798, 0.01101626, 0.        ]], dtype=float32)
```

There is absolutely nothing special about the PyTree returned by `model.init` –
it is just a regular PyTree storing the model's parameters. `params` can be swapped
out with any other PyTree that contains the parameters `model` expects:
```python
zero_params = jax.tree_map(lambda x: jnp.zeros(x.shape), params)
print(zero_params)
model.apply(zero_params, x)
===
Out:
FrozenDict({
    params: {
        layer: {
            bias: Array([0., 0., 0., 0.], dtype=float32),
            kernel: Array([[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]], dtype=float32),
        },
    },
})

Array([[0., 0., 0., 0.]], dtype=float32)
```

Forcing model calls to need the passing of parameters keeps it stateless, and
returning parameters like any other PyTree makes Flax interoperable with JAX
functions – as well as other libraries built on JAX. Essentially, by using Flax
we aren't forced to use other specific frameworks.

If you are used to frameworks like PyTorch, calling models like this feels
unnatural at first. However, I personally quite like it this way – it feels
quite elegant to just pass different parameters to the model to get different
behaviour rather than "load" the weights. A bit subjective and fuzzy, but I like
it. To summarise the difference, if we want to implement $f_\theta(x)$, a
PyTorch module is basically $f_\theta$ (which we can call on $x$). A Flax
module is simply $f$, which needs to be provided parameters $\theta$ before it
can be called on $x$ – or alternatively, we call $f$ on $(\theta, x)$.

All in all, the point of Flax is to **provide a familiar stateful API for
development** whilst **preserving JAX statelessness during runtime**. We can
build our neural network modules in terms of classes and objects, but the final
result is a stateless function `model.apply` that takes in parameters and
inputs, and a PyTree of parameters. 

This is identical behaviour to what we began with (recall our `model_forward`
function at the start of this section), just tied up nicely in a bow! Hence, our
function containing `model.apply` that takes as input our PyTree, can be safely
jit-compiled. The result is the same, a heavily-optimised binary blob we bombard
with data. Nothing changes during runtime, it just makes development easier for
those who prefer reasoning about neural networks in a class-based way whilst
remaining interoperable with, and keeping the performance of JAX

There's a lot more to Flax than this, especially outside the `flax.linen` neural
network API. However, for now, I will move on to a full training loop example
using Flax and **Optax**. I will swing back around to some extra Flax points
later, but I feel some concepts are hard to explain without first showing a
training loop. Without further ado..

## A full training loop with Optax and Flax
- (hard to explain optax without something to target)
- Show the main Optax concepts briefly
    - Stateless optimiser (SGD)
    - Stateful (show optimiser state, and how this must be passed around too)
    - How to call the optimiser?
- Define our configuration options
- Create dataset and dataloader
- Define our VAE model!
    - Briefly describe what a VAE is too
    - Show off the linear model, and how we class condition it.
- Introduce the `create_train_step` and `train_step` pattern
    - Also `create_eval_step` and `eval_step` pattern
- Initialise everything!
- Create main training loop
- Plot some nice samples

Given our changes adding a Flax model, our generic training loop looks something
more like this:

## Nice extra tidbits
- Flax Train State
- Learning Rate schedulers
    - Linear, w/ warmup
- Grad clipping and chaining
- EMA
- Finetuning specific parameters
- Orbax?
    - Just mention? Or show basic use-case?

## Conclusion
- Less ideological, more a practical guide to use JAX + Flax + Optax

