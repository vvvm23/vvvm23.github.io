---
title: "2023 05 16 Jax Post"
date: 2023-05-16T08:41:00+01:00
draft: true
---

> The Plan is Simple:
- Introduction
    - The sprint, what it was, how it got me into Jax
    - Inspired by tutorial series to share my own knowledge
    - The goal of this post
        - Avoid discussing all possibilities here.
        - A practical guide about how you can make your own Jax training loop using modern libraries.
        - No creating everything from scratch - other things already cover this (point to them)
        - Aiming to have some unique takes and explanations, which may involve diving into some fundamental areas (contradicting previous two points but I hope it is valuable)

- Jax Discussion
    - Basic usage of Jax looks a lot like numpy.
        - Show jnp interface
        - First exception is randomness
        - Demonstrate that Jax shouldn't really be used exactly like numpy by a speed comparison.

    - How should we use it? (maybe step by step with code)
        - Simply show what you want to happen in expressive Python.
        - Wrap it in a `jax.jit` decorator to indicate you want this region to be compiled.
        - Let the (python) interpreter execute your function op by op (slowly).
        - This will trace out a computational graph which can be viewed using `jax.make_jaxpr`. Note, this is the raw trace and is not optimised.
        - Pass the traced graph to XLA, where it can be (aggressively) optimised and compiled. Useless ops will be dropped.
        - Next time the function is called, instead of tracing again, the optimised compiled binary blob will be fetched and executed (very fast).
        - This is fantastic if your program is basically running the same function over and over again. Such as in a DL training loop where we just pass different data to the train step.

    - Caveats
        - Should try and jit in the widest possible region to give the most context to the compiler: ideally the entire train step including model forward, model backward passes, and optimser updates.
        - Tracing (by default) uses the **shape** of the input to trace and compile. Hence, changing the shape of the input to the same function will cause the trace and compile stages to happen again. We should aim to keep the input shape the same, or at least limited to a small set of possibilities.
        - In the `jax.jit` regions Python (non-Jax) code will only be executed during tracing and not included in the compiled version. This has a lot of implications, such as conditionals, loops, prints, etc. (expand later)

    - Differences between numpy and jax
        - I feel calling Jax "accelerated numpy" is not giving Jax enough credit. The way of using them is totally different.
        - Jax is slow to dispatch to the accelerator which makes op-by-op (eager) execution much slower than running Numpy on CPU – even with access to crazy fast hardware. This makes this style of execution in Jax untenable apart from debugging.
        - As numpy is intended to be used op-by-op, there is no room for optimisation by the compiler. Hence, the burden on the developer to write performant code and call the fast, heavily optimised, vectorised numpy functions as much as possible, over working at a Python level, which can be orders of magnitude slower.
        - Although of course the developer should think about performance when writing Jax, the burden is reduced by XLA. We are much more free to just write what we want to happen, and rely on XLA to optimise the hell out of everything.
        - Stupid loop example?
        - The above reminds me a bit of programming in Rust, where the developer works with the strict but knowledgeable compiler – kinda like a very serious tango partner. Jax has worse error messages as it stands though

    - Finished my ideological rant, to summarise: **for the final application we want to only run jax ops in jitted regions, where the jitted region is as large as possible, with fixed input shapes. We define what we want to happen which is transformed into a computational graph, and rely on XLA to optimise and compile it.**

    - As jitting functions is such a key concept, it is worth diving deeper into what can and can't be jitted, and how to turn tricky functions into something that can be jitted.
        - Demonstrate the unknown shape error, briefly implied early (as must have shape to trace, not a real array)
        - Shape errors also extend to shapes inside the graph (not just inputs). For example. Show "function with argument-value dependent shapes" example https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
        - In general, it is possible to change dynamicly shaped inputs to something that is static such as through padding or masking.
        - Discuss the pure functions, and how it isn't really pure at a Python level. (implicit arguments that are hidden)
        - Demonstrate the (Python) branching and how only one is traced.
        - Jax conditionals, resulting in both branches being compiled.
        - Similar for Python loops and how they get unrolled
        - A real Jax loop, for example in a diffusion inference loop
        - Which loop to use, trade off between compile time and optimisation potential
        - No in place updates (but XLA may so don't worry about performance, just in place at a Python level makes analysis and transformation difficult)
            - See table https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at
        - All jax ops must have array inputs (unlike numpy) to avoid degradation in performance.
        - However, this doesn't mean the arguments to the jitted function can't be something other than arrays. In fact, they can be any arbitrary PyTree. I won't dive deep into PyTrees here, but here simply treat a PyTree as a nested Python container with arrays at the leaves. This is useful for neural network parameters which lend themselves to be a nested dictionary. Same restrictions apply to shape though: the structure of the nesting and the shape of the array leaves must be the same in order to cache correctly.

    - Another key concept in Jax is the ability to transform functions into other functions. Jax comes with a few inbuilt ones that you will find useful, especially the "A" in Jax which comes from its suite of Autodiff function transformations.
        - It is even possible to write your own custom function transforms, which I will discuss in a later post in this series. 
        - Discuss the autodiff suite of functions
            - `grad` and `value_and_grad`
            - second derivatives by simply nesting
            - aux values and such
        - Touch on `vmap`
            - But I haven't personally found much use for it as I am too used to writing code for batches anyway.
        - Mention `pmap` and other `p` functions! But I'll touch on those in a later post.
            - Jax has pretty excellent support for parallelism which is one of its strengths

- Conclusion
    - I would argue that Jax isn't really accelerated numpy as how you use them is totally different. Expand here to solidify the central point.

---

Recently, I took part in the Huggingface x Google Cloud community sprint which
(despite being a ControlNet sprint) had a very broad scope: involve diffusion
models, use Jax, and use TPUs provided for free. A lot of cool projects came out
of it in a relatively short span of time.

Our project had a pretty ambitious goal: to take my master's dissertation work
on combining step-unrolled denoising autoencoders (loosely adjacent to discrete
diffusion models) with VQ-GAN, porting it all to Jax, then adding support for
text-conditioned generation to it. With this new model, we would train a new
text-to-image model from scratch, a la Dalle-mini.

%%Add unconditional results from paper%%

> Interestingly, Dalle-mini was born out of a previous Huggingface community sprint. So a lot of cool things can come out of these community initiatives!

Unfortunately we didn't manage to achieve our final goal, plagued by a subtle
bug somewhere in the code that meant we never got much out of the model apart
from pretty colours. I wish I could show some cooler outputs, but we simply ran
out of time, despite best efforts by the team.

%%Show model convergence%%

I jumped into Jax by following [an excellent tutorial
series](https://github.com/gordicaleksa/get-started-with-JAX) by [Aleksa
Gordic](https://gordicaleksa.com/). Aleksa prefaces the video with the fact that
he is also just learning Jax. No doubt he is even better now, but I still felt
quite inspired by this attitude: sharing and teaching as you yourself learn.
Hence, I decided that following the sprint, I would channel this spirit and
share what I learnt. And hence, here we are.

%%Screenshot to twitter thing%%

Although it is possible to implement everything in Jax alone – including
manually implementing the optimiser and  model – this isn't really an approach I
really enjoy. During the sprint, we made heavy use of libraries built on top of
Jax such as Flax and Optax. It is definitely valuable to try doing everything
yourself, but if you just want to get started it is similarly worth just leaning
into higher-level frameworks.

Saying that, in this specific blog I will only be covering Jax itself – leaving
creating a fully fledged training loop with higher-level libraries to later
chapters. I initially tried covering everything in unit but the length got far
too much to handle. Related: this won't be a deep dive into Jax, but I hope it
can be a good entry into the Jax ecosystem (young as it may be) and perhaps
share some unique viewpoints useful to those with more experience. Said
viewpoints might contradict the earlier point about "not being a deep dive", but
I suppose we'll just have to live with that.

> If you are curious how doing everything from scratch could be done, I would
> take a look at aforementioned tutorial by Aleksa Gordic, or the official
> tutorials [here](https://jax.readthedocs.io/en/latest/jax-101/index.html).

Without further ado..

## Basic Usage is *Almost* Like NumPy

Jax is a framework initially developed at Google but later open-sourced for
high-performance machine learning research / numerical computing. The name comes
from three of its core components, namely the bringing together of
**J**ust-in-time compilation, **A**utodiff, and **X**LA.

A big draw to Jax is that it shares a similar API to NumPy but can be executed
on fast accelerators such as GPUs and TPUs but written in an accelerator
agnostic fashion. The familiar API also helps train up engineers in Jax – or at
least gets them through the door. Furthermore, it has very good inbuilt support
for multi-device parallelism compared to other frameworks that **could be used
for machine learning** such as PyTorch and Tensorflow.

Although definitely intended to support machine learning research, to me it
appears to have a less strong bias towards machine learning and is more readily
applied to other domains. This is somewhat akin to NumPy which is a general
purpose array manipulation library, but I believe the way you should use Jax is
very different to NumPy, despite initial appearances.

Specifically, if NumPy is about manipulating arrays operation by operation, Jax
is about **defining computational graphs and letting Jax optimise it**. In other
words, defining what you want to happen and letting Jax do the heavy lifting in
making it run fast. In NumPy, the burden is on the developer to optimise
everything by calling into fast and heavily optimised functions, and avoid slow
Python land as much as possible. However, this extra burden does garner a degree
of extra flexibility over the more rigid Jax land. But crucially, in a lot of
machine learning applications we don't need such flexibility.

Enough ideological rants, let's see that friendly Jax Numpy API, beginning by
initialising a few arrays.

```python
import jax
import jax.numpy as jnp

import numpy as np

L = [0, 1, 2, 3]
x_np = np.array(L, dtype=np.int32)
x_jnp = jnp.array(L, dtype=jnp.int32)

x_np, x_jnp
===
Out: (Array([0, 1, 2, 3], dtype=int32), array([0, 1, 2, 3], dtype=int32))
```
> Note, you may have seen in older tutorials the line `import jax.numpy as np`.
> This is no longer the convention and prior suggestions to do so will remain a
> stain on history.

Frighteningly similar right? The `jax.numpy` interface closely mirrors that of
`numpy`, which means pretty much anything we could do `numpy` we can also do in
`jax.numpy` using very similar functions.

```python
x1 = x_jnp*2
x2 = x_jnp+1
x3 = x1 + x2

x1, x2, x3
===
Out: (Array([0, 2, 4, 6], dtype=int32),
 Array([1, 2, 3, 4], dtype=int32),
 Array([ 1,  4,  7, 10], dtype=int32))
```

```python
jnp.dot(x1, x2), jnp.outer(x1, x2)
===
Out: (Array(40, dtype=int32),
 Array([[ 0,  0,  0,  0],
        [ 2,  4,  6,  8],
        [ 4,  8, 12, 16],
        [ 6, 12, 18, 24]], dtype=int32))
```

All of this should look familiar to you if you have used NumPy before. I won't
bore you to death by enumerating functions. The first interesting difference is
how Jax handles randomness. In NumPy, to generate a random array from the
uniform distribution, we can simply do:
```python
random_np = np.random.random((5,))
random_np
===
Out: array([0.58337985, 0.87832186, 0.08315021, 0.16689551, 0.50940328])
```

In Jax it works differently. A key concept in Jax is that functions in it are
**pure**. This means that given the same input they will always return the same
output, and do not modify any global state from within the function. Using
random number generation that modifies some global psuedorandom number generator
(PRNG) clearly violates both principles. Therefore, we have to handle randomness
in a stateless way by manually passing around the PRNG key and splitting it to
create new random seeds. This has the added benefit of making randomness in code
more reproducible – ignoring accelerator side stochasticity – as in Jax we are
forced to handle fixed seeds by default. Let's see what that looks like:
```python
seed = 0x123456789 # some integer seed. In hexadecimal just to be ✨✨
key = jax.random.PRNGKey(seed) # create the initial key
key, subkey = jax.random.split(key) # split the key
random_jnp = jax.random.uniform(subkey, (5,)) # use `subkey` to generate, `key` can be split into more subkeys later.
random_jnp
===
Out: Array([0.2918682 , 0.90834665, 0.13555491, 0.08107758, 0.9746183 ], dtype=float32)
```

It is important to not reuse the same key if you want each random op to produce different outputs:
```python
jax.random.normal(key, (2,)), jax.random.normal(key, (2,))
===
Out: (Array([-0.67039955,  0.02259737], dtype=float32),
 Array([-0.67039955,  0.02259737], dtype=float32))
```

You may be pleased to know that if we want to generate `N` random arrays, we don't need to call `jax.random.split` `N` times. Pass the number of keys you want to the function:
```python
key, *subkeys = jax.random.split(key, 5)
[jax.random.normal(s, (2,2)) for s in subkeys]
===
Out: [Array([[ 1.0308125 , -0.07533383],
        [-0.36027843, -1.270425  ]], dtype=float32),
 Array([[ 0.34779412, -0.11094793],
        [ 1.0509511 ,  0.52164143]], dtype=float32),
 Array([[ 1.5565109 , -0.9507161 ],
        [ 1.4706124 ,  0.25808835]], dtype=float32),
 Array([[-0.5725152 , -1.1480215 ],
        [-0.6206856 , -0.12488112]], dtype=float32)]
```

That's kinda interesting, but I am not seeing a great deal of pull towards Jax over NumPy so far. It gets more concerning when we start timing the functions:
```python
x1_np, x2_np = np.asarray(x1), np.asarray(x2)
%timeit x1_np @ x2_np
%timeit x1 @ x2
===
Out: 1.17 µs ± 6.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
7.27 µs ± 1.68 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```
The Jax version of the above multiply is about 6-7 times slower, what gives?

It goes back to my earlier point that NumPy is intended for array manipulation in an op-by-op (or eager) fashion, whereas Jax is all about defining graphs and letting Jax optimise it for you. By executing Jax functions eagerly like NumPy, we leave no room for optimisation and, due to extra Jax overhead, we get a slower function. Bluntly, if you are using Jax like this, you have done something wrong.

So, how do we get Jax to go fast?