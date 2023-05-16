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