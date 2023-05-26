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
- The ways of defining a module.
    - regular and compact representation
- What does that give us?
    - easy parameter init as a PyTree
    - easy way to call the model (whilst remaining stateless)
    - easier to reason about for the developer (class based)
- Model itself is essentially a hollow shell: loosely associating parameters with some ops. 
    - Essentially a helper object. We can imagine a function $f_\Theta(x)$.
    - In PyTorch, a module is $f_\Theta$ that we can apply $x$ to.
    - In Flax, a module is literally $f$, which we apply both $\Theta$ and $x$ to.
    - It makes it quite easy to swap out params.
    - Show that we can pass any pytree with the correct structure.
- params as a PyTree makes it interoperate perfectly with not only JAX, but other libraries built on top of JAX.
    - This modularity is quite common in the JAX ecosystem, we aren't locked into something (typically) if we pick one library.
- The final result is the same. We get params and ops that is passed to a function. This gets traced and compiled as before. We get an optimised compiled function.
- There is a way to bind parameters to a model, and yield an interactive model like $f_\Theta$. However, can't train with this, it is a static model.
- Flax comes with a bunch of other layers inbuilt. I won't enumerate them all, but all the usual culprits are there.
    - Keep in mind though, some default initialisers are different which could be an issue if you are porting models from other frameworks to Flax.
- There is a lot more to Flax than this, but enough for now. Main takeaway are:
    - During dev time, Flax helps the developer reason about neural networks in an object-orientated way whilst remaining functional during runtime.
    - During runtime, `flax.linen` is a helper for creating stateless shells that build PyTrees of parameters and loosely associate said parameters with JAX operations.
    - Statelessness is important to allow Flax to interoperate with JAX and other libraries built on JAX, but also kinda neat.

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

