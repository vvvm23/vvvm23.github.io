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
        - Show jnp interface X
        - First exception is randomness X
        - Demonstrate that Jax shouldn't really be used exactly like numpy by a speed comparison. X

    - How should we use it? (maybe step by step with code)
        - Simply show what you want to happen in expressive Python. X
        - Wrap it in a `jax.jit` decorator to indicate you want this region to be compiled. X
        - Let the (python) interpreter execute your function op by op (slowly). X
        - This will trace out a computational graph which can be viewed using `jax.make_jaxpr`. Note, this is the raw trace and is not optimised. X
        - Pass the traced graph to XLA, where it can be (aggressively) optimised and compiled. Useless ops will be dropped. X
        - Next time the function is called, instead of tracing again, the optimised compiled binary blob will be fetched and executed (very fast). X
        - This is fantastic if your program is basically running the same function over and over again. Such as in a DL training loop where we just pass different data to the train step. X

    - Caveats
        - Should try and jit in the widest possible region to give the most context to the compiler: ideally the entire train step including model forward, model backward passes, and optimser updates. X
        - Tracing (by default) uses the **shape** of the input to trace and compile. Hence, changing the shape of the input to the same function will cause the trace and compile stages to happen again. We should aim to keep the input shape the same, or at least limited to a small set of possibilities. X
        - In the `jax.jit` regions Python (non-Jax) code will only be executed during tracing and not included in the compiled version. This has a lot of implications, such as conditionals, loops, prints, etc. (expand later)

    - Differences between numpy and jax
        - I feel calling Jax "accelerated numpy" is not giving Jax enough credit. The way of using them is totally different. X
        - Jax is slow to dispatch to the accelerator which makes op-by-op (eager) execution much slower than running Numpy on CPU – even with access to crazy fast hardware. This makes this style of execution in Jax untenable apart from debugging. X
        - As numpy is intended to be used op-by-op, there is no room for optimisation by the compiler. Hence, the burden on the developer to write performant code and call the fast, heavily optimised, vectorised numpy functions as much as possible, over working at a Python level, which can be orders of magnitude slower. X
        - Although of course the developer should think about performance when writing Jax, the burden is reduced by XLA. We are much more free to just write what we want to happen, and rely on XLA to optimise the hell out of everything. X
        - Stupid loop example? X
        - The above reminds me a bit of programming in Rust, where the developer works with the strict but knowledgeable compiler – kinda like a very serious tango partner. Jax has worse error messages as it stands though X

    - Finished my ideological rant, to summarise: **for the final application we want to only run jax ops in jitted regions, where the jitted region is as large as possible, with fixed input shapes. We define what we want to happen which is transformed into a computational graph, and rely on XLA to optimise and compile it.**

    - As jitting functions is such a key concept, it is worth diving deeper into what can and can't be jitted, and how to turn tricky functions into something that can be jitted.
        - Shape errors also extend to shapes inside the graph (not just inputs). For example. Show "function with argument-value dependent shapes" example https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html X
        - In general, it is possible to change dynamicly shaped inputs to something that is static such as through padding or masking. X
        - Discuss the pure functions, and how it isn't really pure at a Python level. (implicit arguments that are hidden) X
        - Demonstrate the (Python) branching and how only one is traced. X
        - Jax conditionals, resulting in both branches being compiled.
        - Similar for Python loops and how they get unrolled X
        - A real Jax loop, for example in a diffusion inference loop X
        - Which loop to use, trade off between compile time and optimisation potential X
        - No in place updates (but XLA may so don't worry about performance, just in place at a Python level makes analysis and transformation difficult) X
            - See table https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at
        - All jax ops must have array inputs (unlike numpy) to avoid degradation in performance. X
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

Another small difference is that Jax does not support inplace operations:
```python
x1[0] = 5
===
Out: 
TypeError                                 Traceback (most recent call last)

<ipython-input-25-e0318c4eb619> in <cell line: 1>()
----> 1 x1[0] = 5

/usr/local/lib/python3.10/dist-packages/jax/_src/numpy/array_methods.py in _unimplemented_setitem(self, i, x)
    261          "or another .at[] method: "
    262          "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html")
--> 263   raise TypeError(msg.format(type(self)))
    264 
    265 def _operator_round(number: ArrayLike, ndigits: Optional[int] = None) -> Array:

TypeError: '<class 'jaxlib.xla_extension.ArrayImpl'>' object does not support item assignment. JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
```
Like the error message says, Jax arrays are **immutable**, hence the same issue
applies to other inplace ops like `+=`, `*=`, and friends. Also like the error
message says, we can use the `at` property on Jax arrays to perform functionally pure equivalents.
```python
x1_p999 = x1.at[0].add(999)
x1, x1_p999
===
Out: (Array([0, 2, 4, 6], dtype=int32), Array([999,   2,   4,   6], dtype=int32))
```

> Applying `x1 += 5` and similar *does* work, but under-the-Python-hood this is
just `x1 = x1 + 5` anyway. It just creates a new array and hence is still
immutable.

Jax functions also only accept array inputs. This is contrast to NumPy that will
happily accept Python lists. Jax chooses to do this to avoid silent degradation
in performance by just erroring instead.

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

So, how do we get Jax to go fast? By making use of XLA.

## Enter `jax.jit`

The reason why the earlier functions were so slow is that Jax is dispatching to
the accelerator one operation at a time. The intended way to use Jax is to
compile multiple operations – ideally nearly all operations – together using
XLA. To indicate which region to compile together, we can pass the function we
want to compile to `jax.jit` or use the `@jax.jit` decorator. The function will
not be compiled immediately, but rather upon first call – hence the name "Just
in time compilation".

During this first call, the shapes of the input arrays will be used to trace out
a computational graph, stepping through the function with the Python interpreter
and executing all operations one-by-one, recording what happens as we go. This
intermediate representation can be given to XLA and subsequently compiled,
optimised, and cached. This cache will be retrieved if the same function is
called with the same input array shapes and dtype, skipping the tracing and
compilation process, calling the compiled binary blob directly.

Let's see it in action:

```python
def fn(W, b, x):
    return x @ W + b

key, w_key, b_key, x_key = jax.random.split(key, 4)
W = jax.random.normal(w_key, (4, 2)),
b = jax.random.uniform(b_key, (2,))
x = jax.random.normal(x_key, (4,))

print("`fn` time")
%timeit fn(W, b, x)

print("`jax.jit(fn)` first call time")
jit_fn = jax.jit(fn)
%time jit_fn(W, b, x)

print("`jit_fn` time")
%timeit jit_fn(W, b, x)
===
Out: 
`fn` time
26.1 µs ± 1.56 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

`jit_fn` first call (warmup) time
CPU times: user 35.8 ms, sys: 38 µs, total: 35.9 ms
Wall time: 36.3 ms

`jit_fn` time
7.62 µs ± 1.88 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

Like expected, the first call will take much longer than the subsequent calls.
It is important to exclude the first call from any benchmarking for this reason.
Also as expected, we see that even for this simple example the compiled version
of the function executes far quicker than the op-by-op function.

It is possible to view the traced graph as a `jaxpr` using `jax.make_jaxpr` on
an input function. Though, somewhat hard to read once the functions grow more complex.
```python
jax.make_jaxpr(fn)(params, x)
===
Out: { lambda ; a:f32[4,2] b:f32[2] c:f32[4]. let
    d:f32[2] = dot_general[dimension_numbers=(([0], [0]), ([], []))] c a
    e:f32[2] = add d b
  in (e,) }
```

And also the compiled version of the function, which is even more difficult to read.
```python
print(jax.jit(fn).lower(params, x).compile().as_text())
===
HloModule jit_fn, entry_computation_layout={(f32[4,2]{1,0},f32[2]{0},f32[4]{0})->f32[2]{0}}, allow_spmd_sharding_propagation_to_output={true}

%fused_computation (param_0.1: f32[2], param_1.1: f32[4], param_2: f32[4,2]) -> f32[2] {
  %param_1.1 = f32[4]{0} parameter(1)
  %param_2 = f32[4,2]{1,0} parameter(2)
  %dot.0 = f32[2]{0} dot(f32[4]{0} %param_1.1, f32[4,2]{1,0} %param_2), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_name="jit(fn)/jit(main)/dot_general[dimension_numbers=(((0,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="<ipython-input-4-04cd19da0726>" source_line=2}
  %param_0.1 = f32[2]{0} parameter(0)
  ROOT %add.0 = f32[2]{0} add(f32[2]{0} %dot.0, f32[2]{0} %param_0.1), metadata={op_name="jit(fn)/jit(main)/add" source_file="<ipython-input-4-04cd19da0726>" source_line=2}
}

ENTRY %main.6 (Arg_0.1: f32[4,2], Arg_1.2: f32[2], Arg_2.3: f32[4]) -> f32[2] {
  %Arg_1.2 = f32[2]{0} parameter(1), sharding={replicated}
  %Arg_2.3 = f32[4]{0} parameter(2), sharding={replicated}
  %Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={replicated}
  ROOT %fusion = f32[2]{0} fusion(f32[2]{0} %Arg_1.2, f32[4]{0} %Arg_2.3, f32[4,2]{1,0} %Arg_0.1), kind=kOutput, calls=%fused_computation, metadata={op_name="jit(fn)/jit(main)/add" source_file="<ipython-input-4-04cd19da0726>" source_line=2}
}
```

A more explicit and silly example is below:
```python
def stupid_fn(x):
  y = jnp.copy(x)
  for _ in range(1000):
    x = x * x
  return y

print("`stupid_fn` time")
%time stupid_fn(x)

print("`jit_stupid_fn` first call")
jit_stupid_fn = jax.jit(stupid_fn)
%time jit_stupid_fn(x)

print("`jit_stupid_fn` time")
%timeit jit_stupid_fn(x)
===
Out: 
`stupid_fn` time
CPU times: user 17.2 ms, sys: 0 ns, total: 17.2 ms
Wall time: 17.6 ms

`jit_stupid_fn` first call
CPU times: user 1.03 s, sys: 5.79 ms, total: 1.04 s
Wall time: 1.09 s

`jit_stupid_fn` time
5.6 µs ± 1.67 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

In the function, we copy the input to variably `y`, then multiply the input with
itself 1,000 times. Finally, we simply return `y`, making the multiplications
totally pointless. In the non-jit version, the program will happily and
pointlessly perform the multiplication. Ignorance is bliss. On first call to the
jit function, again we will step through all the multiplications as Jax traces
out the computational graph. However, the compiled version used on subsequent
calls will be blazing fast, as XLA sees the multiplications are not needed to
obtain the final output. We can actually see this by printing the `jaxpr`:
```python
jax.make_jaxpr(stupid_fn)(x)
===
Out: { lambda ; a:f32[4]. let
    b:f32[4] = copy a
    c:f32[4] = mul a a
    d:f32[4] = mul c c
    e:f32[4] = mul d d
    f:f32[4] = mul e e
    ... <truncated>
    bmh:f32[4] = mul bmg bmg
    bmi:f32[4] = mul bmh bmh
    bmj:f32[4] = mul bmi bmi
    bmk:f32[4] = mul bmj bmj
    bml:f32[4] = mul bmk bmk
    bmm:f32[4] = mul bml bml
    _:f32[4] = mul bmm bmm
  in (b,) }
```
Which shows all 1,000 multiplications, and comparing it with the compiled version:
```python
print(jax.jit(stupid_fn).lower(x).compile().as_text())
===
Out: 
HloModule jit_stupid_fn, entry_computation_layout={(f32[4]{0})->f32[4]{0}}, allow_spmd_sharding_propagation_to_output={true}

ENTRY %main.2 (Arg_0.1: f32[4]) -> f32[4] {
  %Arg_0.1 = f32[4]{0} parameter(0), sharding={replicated}
  ROOT %copy = f32[4]{0} copy(f32[4]{0} %Arg_0.1)
}
```
Which only contains a single copy operation. Experiment with the above code
blocks yourself by changing the number of iterations in the loop. You will find
that the time to execute the original function will increase with number of
iterations. Additionally, the time to trace the graph on first call to the jit
function will also increase, as this happens using the Python interpreter.
However, the time to execute the compiled version on subsequent calls will not
increase in a meaningful way.

The above is a contrived example, but demonstrates a critical point: **we can
let XLA do a lot of the heavy lifting for us optimisation-wise.** This is
different to other frameworks that execute eagerly, where the above code would
happily execute extremely pointlessly. This isn't really a fault of the
framework as eager execution has a ton of other benefits, but demonstrates the
point that compiling our functions using XLA can help optimise our code in ways
we didn't know about, or could reasonably anticipate. What exact optimisations
XLA applies is a topic outside the scope of this blog, however one example is
that the earlier statement about Jax arrays not allowing in-place operations
results in no potential performance loss. This is because XLA can identify cases
where it can replace operations with in-place equivalents. So basically, don't
sweat it if you were worried earlier about not being able to do stuff in-place.

Secondly, in order to let XLA do the best job it can, **`jax.jit` needs to be
used in the widest possible context**. For example, (again contrived) if we had
only jit compiled the multiplication, XLA would be unaware that the outermost
loop was unnecessary and could not optimise it out – it is simply outside the
region to be compiled. A concrete machine learning example would be wrapping the
entire training step – forward, backwards and optimiser step – in `jax.jit`.

It turns out most machine learning applications can be expressed in this way:
one monolithic compiled function that we throw data and model parameters at. In
the original Jax paper, they say "The design of JAX is informed by the
observation that ML work- loads are typically dominated by PSC
(pure-and-statically-composed) subroutines" which lends itself well to this
compilation process. Even functions that are seemingly not static can be
converted into a static form, for example padding sequences in language modeling
tasks or rewriting our functions in clever ways.

Although eager mode execution is very useful for development work, once
development is done there is less benefit to eager execution over heavily
optimised binary blobs, hungry for our data. However, such optimisation relies
on said pureness and staticness, which must be enforced in order to jit-compile
our functions. 

## Jit needs static shapes

The biggest blocker to jit compiling functions is that **all arrays to have
static shapes**. That is to say, given the **shapes** and shapes alone of the
function inputs, it should be possible to determine the shape of all other
variables in the traced graph at compile time.

Take for example the following function, that given an integer `length` returns
an array filled with the value `val`:
```python
def create_filled(val, length):
  return jnp.full((length,), val)

print(create_filled(1.0, 5))
print(create_filled(2, 2))

jit_create_filled = jax.jit(create_filled)
jit_create_filled(2, 5)
===
Out: [1. 1. 1. 1. 1.]
[2 2]

---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

<ipython-input-13-0ecd13642388> in <cell line: 8>()
      6 
      7 jit_create_filled = jax.jit(create_filled)
----> 8 jit_create_filled(2, 5)

    [... skipping hidden 12 frame]

3 frames

/usr/local/lib/python3.10/dist-packages/jax/_src/core.py in canonicalize_shape(shape, context)
   2037   except TypeError:
   2038     pass
-> 2039   raise _invalid_shape_error(shape, context)
   2040 
   2041 def canonicalize_dim(d: DimSize, context: str="") -> DimSize:

TypeError: Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>,).
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function create_filled at <ipython-input-13-0ecd13642388>:1 for jit. This concrete value was not available in Python because it depends on the value of the argument length.
```

In eager execution, the function returns what we expected. However, when tracing
the jit version of the function we encounter an error. This is because when
tracing the `jnp.ones` function will receive only a traced, zero-dimensional
array which only contains information about the shape and dtype. It is therefore
impossible to trace the output array as the shape is not known at compile time.

We can resolve this issue by using an extra argument in `jax.jit` named
`static_argnums. This specifies which arguments to **not** trace and just treat
it as a regular Python value at compile time. In the `jaxpr` graph, the `length`
argument to our Python-level function essentially becomes a constant in the graph:
```python
jit_create_filled = jax.jit(create_filled, static_argnums=(1,))
print(jit_create_filled(2, 5))
print(jit_create_filled(1., 10))

print(jax.make_jaxpr(create_filled, static_argnums=(1,))(2, 5))
print(jax.make_jaxpr(create_filled, static_argnums=(1,))(1.6, 10))
===
Out: [2 2 2 2 2]
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

{ lambda ; a:i32[]. let
    b:i32[5] = broadcast_in_dim[broadcast_dimensions=() shape=(5,)] a
  in (b,) }
{ lambda ; a:f32[]. let
    b:f32[10] = broadcast_in_dim[broadcast_dimensions=() shape=(10,)] a
  in (b,) }
```

As the shape is a constant in the graph now, each time a different `length` is
passed to the function the function will need to be recompiled. Hence, this
approach only really works if the number of values `length` will take is very
limited, otherwise we will be constantly compiling different graphs.

Make no mistake, even though the Python-level function is identical, the
underlying binaries that are called for different inputs are completely
different. We've basically turned the caching from matching on function and
input shapes, to matching on function, input shapes, and also the **value** of
our static arguments.

A different example now: let's define a function that takes in an input array `x` and
boolean mask `mask` with the same shape as `x` and returns a new array with
masked positions set to a large negative number.

```python
def mask_tensor(x, mask):
  x = x.at[mask].set(-100.)
  return x

key, x_key, mask_key = jax.random.split(key, 3)
x = jax.random.normal(x_key, (4,4))
mask = jax.random.uniform(mask_key, (4,4)) < 0.5

print("calling eager function")
print(mask_tensor(x, mask))

print("calling compiled function")
jit_mask_tensor = jax.jit(mask_tensor)
jit_mask_tensor(x, mask)
===
Out: calling eager function
[[-3.8728207e-01 -1.3147168e+00 -2.2046556e+00  4.1792620e-02]
 [-1.0000000e+02 -1.0000000e+02 -8.2206033e-02 -1.0000000e+02]
 [ 2.1814612e-01  9.6735013e-01  1.3497342e+00 -1.0000000e+02]
 [-8.7061942e-01 -1.0000000e+02 -1.0000000e+02 -1.0000000e+02]]
calling compiled function

---------------------------------------------------------------------------

NonConcreteBooleanIndexError              Traceback (most recent call last)

<ipython-input-23-2daf7923c05b> in <cell line: 14>()
     12 print("calling compiled function")
     13 jit_mask_tensor = jax.jit(mask_tensor)
---> 14 jit_mask_tensor(x, mask)

    [... skipping hidden 12 frame]

5 frames

/usr/local/lib/python3.10/dist-packages/jax/_src/numpy/lax_numpy.py in _expand_bool_indices(idx, shape)
   4297       if not type(abstract_i) is ConcreteArray:
   4298         # TODO(mattjj): improve this error by tracking _why_ the indices are not concrete
-> 4299         raise errors.NonConcreteBooleanIndexError(abstract_i)
   4300       elif _ndim(i) == 0:
   4301         raise TypeError("JAX arrays do not support boolean scalar indices")

NonConcreteBooleanIndexError: Array boolean indices must be concrete; got ShapedArray(bool[4,4])
```

Executing the function in eager mode works as expected. However, the shape of
intermediate variables cannot be known given knowledge of the input shapes
alone, but rather depends on the number of elements in `mask` that are `True`.
Therefore, we cannot compile the function as not all shapes are static.

Additionally, we can't simply use `static_argnum` as `mask` itself is not
hashable and hence can't be used to match calls to caches. Furthermore, even if
it could the number of possible values of `mask` is too high. To handle all
possibiltiies, we would need to compile `2**16` or 65,536 graphs.

Often though, we can rewrite the function to perform the same action and with
static shapes:

```python
def mask_tensor(x, mask):
  x = ~mask * x - mask*100.
  return x

print("calling eager function")
print(mask_tensor(x, mask))

print("calling compiled function")
jit_mask_tensor = jax.jit(mask_tensor)
print(jit_mask_tensor(x, mask))
===
calling eager function
[[   1.012518   -100.           -0.8887863  -100.        ]
 [-100.         -100.         -100.            1.5008001 ]
 [-100.           -0.6636745     0.57624763   -0.94975847]
 [   1.1513114  -100.            0.88873196 -100.        ]]
calling compiled function
[[   1.012518   -100.           -0.8887863  -100.        ]
 [-100.         -100.         -100.            1.5008001 ]
 [-100.           -0.6636745     0.57624763   -0.94975847]
 [   1.1513114  -100.            0.88873196 -100.        ]]
```

All intermediate shapes will be known at compile time. To break it down, we
multiply `x` by zero where `mask` is True, and by one where it is `False`. We
then add a new array that is zero where `mask` is `False` and `-100` where
`mask` is `True`. At this point we have two arrays with concrete shapes. Adding
them together yields the correct result, which is similarly concrete.

## Limit the number of possible input shapes

A related case that can "kinda" be jit compiled is where shapes can be
determined at compile time but the shapes of the inputs change a lot. As we
retrieve cached compiled functions by looking at which function was called and
the shape of the inputs, this will result in a lot of compiling. This makes
sense, as the graph itself is optimised for a specific static shape, but will
result in silent slowdowns.

```python
import random

def cube(x):
  return x*x*x

def random_shape_test(fn):
  length = random.randint(1, 1000)
  return fn(jnp.empty((length,)))

print("random length eager time:")
%timeit -n1000 random_shape_test(cube)

jit_cube = jax.jit(cube)
jit_cube(x1)

print("fixed length compiled time:")
%timeit -n1000 jit_cube(x1)

print("random length compiled time:")
%timeit -n1000 random_shape_test(jit_cube)
===
Out:
random length eager time:
The slowest run took 43.13 times longer than the fastest. This could mean that an intermediate result is being cached.
6.12 ms ± 8.37 ms per loop (mean ± std. dev. of 7 runs, 1000 loops each)

fixed length compiled time:
7.31 µs ± 241 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)

random length compiled time:
The slowest run took 53.37 times longer than the fastest. This could mean that an intermediate result is being cached.
4.55 ms ± 6.11 ms per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

Therefore, we should try our best to limit the number of shapes that our jitted
functions will take as input. Common examples include padding sequences to a
single length, or setting `drop_last=True` on data loaders to avoid different
numbers of examples in a batch.

## Functional Purity and Side Effects

Jax transformations and compilation are designed to only work on pure Python
functions. Roughly speaking, a **functionally pure function is one where given
the same inputs, it will always produce the same outputs, and does not have any
observable side effects.**

For example, see this example where the output of `fn` relies not only on `x`
but also on `shift`, which we change between function calls:
```python
shift = -1.0
x1 = jnp.array([0, 1, 2])
x2 = jnp.array([0, -1, 0])
x3 = jnp.array([0, 1, 2, 3])
def fn(x):
  return x + shift

print(fn(x1))

shift = 1.0
print(fn(x2))
print(fn(x3))

shift = -1.0
jit_fn = jax.jit(fn)

print(jit_fn(x1))

shift = 1.0
print(jit_fn(x2))

print(jit_fn(x3))
===
Out:
[-1.  0.  1.]
[1. 0. 1.]
[1. 2. 3. 4.]

[-1.  0.  1.]
[-1. -2. -1.]
[1. 2. 3. 4.]
```
The eager mode calls (the first three) represent our ground truth, the last
three are outputs of the jit function using the same inputs and global `shift
value`. In the jit function, the first call of a given shape (when we trace)
will use the correct current global shift value. If we call again and Jax finds
a cached function, it won't look at the new global shift but rather just
execute the compiled code directly - the one that has baked in the old value in
the graph as a constant. However, if tracing is triggered again (such as with a
different input shape) the correct `shift` will be used.

This is what they mean by "Jax transformations and compilation are **designed**
to only work on pure functions". They can still be applied to impure but the
behaviour of the function will diverge from the Python interpreter when tracing
is skipped and the compiled function is used directly. Another example is one
that involves a `print` function:
```python
def fn(x):
  print("called identity function")
  return x

jit_fn = jax.jit(fn)

print("called `jit_fn(0.5)`")
_ = jit_fn(0.5)
print("called `jit_fn(1.0)`")
_ = jit_fn(1.0)
print("called `jit_fn([-1, 1])`")
_ = jit_fn(jnp.array([-1, 1]))
===
Out:
called `jit_fn(0.5)`
called identity function

called `jit_fn(1.0)`

called `jit_fn([-1, 1])`
called identity function
```
Again, **whenever tracing is triggered, the behaviour is the same as Python**,
but whenever the cached function is used, behaviour diverges. This is again
impure as `print` is a side effect.

What about when the global we are using is also a Jax array?
```python
b = jnp.array([1,2,3])

def fn(x):
  return x + b

jit_fn = jax.jit(fn)

x = jnp.array([1,2,3])
print(jit_fn(x))

b = jnp.array([0,0,0])
print(jit_fn(x))
===
[2 4 6]
[2 4 6]
```
Again, as the input shape of `x` hasn't changed, the compiled version will be
used, hence the value of `b` in the function won't be updated. However, `b` is
actually a variable in the graph, unlike our previous example modifying
`shift`. Jax maintains functional purity in the compiled function by adding `b`
as an **implicit argument** in the traced graph. Hence, the graph is
functionally pure, however `b` is essentially a constant for us as we have no
way of modifying this implicit argument at a Python-level without recompiling.

Generally speaking, the **final compiled function** is pure. However, the
Python-level function we created isn't necessarily pure, and `jax.jit` can
still be applied, but needs care. I would summarise the caveats as follows
though:
- Code that does not manipulate Jax arrays will not be traced and only called
  during tracing itself (as the Python interpreter steps through the function,
  and evaluates the code like any other Python code). Examples of this include
  `print` statements and setting Python level variables, as well as
  Python-level conditionals and loops.
- Code that does manipulate Jax arrays **but** the Jax array is not an argument
  to the Python function (perhaps it is global, relative to the function) we
  are jit compiling will be traced, but those variables in the graph will take
  whatever value they had at **compile-time** and become implicit arguments to
  the traced graph.

I feel both of these impure cases still have value. For example, the first is
nice when debugging shape issues (the first call will still print the shapes!)
or perhaps disabling parts of the function using some global configuration
object:
```python
config = dict(relu=False)

@jax.jit
def fn(W, x):
    y = x @ W
    if config['relu']:
        y = jax.nn.relu(y)
    return y

W, x = jnp.ones((2,2)), jnp.ones((2,))
jax.make_jaxpr(fn)(W, x)
===
Out:
{ lambda ; a:f32[2,2] b:f32[2]. let
    c:f32[2] = pjit[
      jaxpr={ lambda ; d:f32[2,2] e:f32[2]. let
          f:f32[2] = dot_general[dimension_numbers=(([0], [0]), ([], []))] e d
        in (f,) }
      name=fn
    ] a b
  in (c,) }
```
You can see in the `jaxpr` that only the `dot_general` is present in the graph.
The `relu` function was not traced as the Python interpreter didn't execute the
body of the `if` statement, and hence didn't add it to the graph. It is
important to emphasise that **only a single conditional branch was compiled**:
there is no branching in the final graph.

> Arguably, there is a case for using `static_argnums` if you expect to use
> both options in a single execution of your program. However if your `config`
> object won't change, I feel the above pattern is fine!

> It is possible to add conditionals in the compiled function. However,
> Python-level conditionals are only used when tracing. Special functions
> (shown later) must be used to add conditionals in the final compiled
> function.

The second point can be useful if we have some object we know won't change, for
example a pretrained machine learning model that we just want to run fast
inference on:
```python
bert = ... # some pretrained Jax BERT model that we can call

@jax.jit
def fn(x):
  return bert(x)
```
The above would work, but changes to `bert` would not be reflected in the
compiled function until the shape of `x` changes. We could even set `bert` to
be `None` following the first call and `fn` would still work, provided we used
the same input shape.

In general, I feel the emphasis on making things functionally pure is a bit
overstated in Jax. In my (perhaps misinformed) opinion, it is better to simply
understand the difference of trace-time and compiled behaviour, and when they
will be triggered. Python is ridiculously expressive, and making use of that is
part of the power of Jax, so it would be a shame to needlessly restrict that.

## Conditionals and Loops in Compiled Functions

I hope now that you have developed a bit of an intuition into the difference
between trace-time and compiled behaviour. But if not, here is a summary:
- Tracing occurs when a jit compiled function encounters a set of input shapes
  and static argument values that it hasn't encountered yet. In such cases, Jax
  relies on the Python interpreter to step through the function. All normal
  Python rules apply in this case. The traced graph will **contain traceable
  operations that were encountered during this specific instance of tracing**.
- Calling the compiled version occurs when a jit compiled function is called
  and the set of input shapes and static argument values match one in the
  cache. In such cases, **behaviour is simply calling the compiled function and
  nothing more**.

This behaviour is powerful, as it allows us to define what we want to happen in
expressive Python, and rely on fast, optimised code for the actual execution.
However, it does come with some issues:
- We can only trace one conditional path per input shapes and static value
  combination.
- As tracing steps through op-by-op, loops will simply be unrolled, rather than
  being loops in the final compiled function.

Sometimes these properties are attractive. The first can be used to simply
disable branches we don't care about, almost like compile time flags in C. The
second is useful for small numbers of loop iterations where cross-iteration
dependencies can be optimised. However, sometimes this works against us.

We've already seen one example of this, recall `stupid_fn`:
```python
def stupid_fn(x):
  y = jnp.copy(x)
  for _ in range(1000):
    x = x * x
  return y

jax.make_jaxpr(stupid_fn)(jnp.array([1.1, -1.1]))
===
Out:
Out: { lambda ; a:f32[4]. let
    b:f32[4] = copy a
    c:f32[4] = mul a a
    d:f32[4] = mul c c
    e:f32[4] = mul d d
    f:f32[4] = mul e e
    ... <truncated>
    bmh:f32[4] = mul bmg bmg
    bmi:f32[4] = mul bmh bmh
    bmj:f32[4] = mul bmi bmi
    bmk:f32[4] = mul bmj bmj
    bml:f32[4] = mul bmk bmk
    bmm:f32[4] = mul bml bml
    _:f32[4] = mul bmm bmm
  in (b,) }
```
I truncated this earlier, but it is egregiously long in terminal. During
tracing the entire loop gets unrolled. Not only is this annoying to look at,
but it makes optimising the graph take a long time, which makes the first call
to the function so long. Jax isn't aware we are in a for-loop context, it
simply just takes the operations and adds it to the graph.

Luckily, Jax exposes control flow primitives as part of its `jax.lax`
submodule:
```python
def less_stupid_fn(x):
    y = jnp.copy(x)
    x = jax.lax.fori_loop(start=0, stop=1000, body_fun=lambda i, x: x * x, init_val=x)
    return y
jax.make_jaxpr(less_stupid_fn)(jnp.array([1.1, -1.1]))
===
Out:
{ lambda ; a:f32[2]. let
    b:f32[2] = copy a
    _:i32[] _:f32[2] = scan[
      jaxpr={ lambda ; c:i32[] d:f32[2]. let
          e:i32[] = add c 1
          f:f32[2] = mul d d
        in (e, f) }
      length=1000
      linear=(False, False)
      num_carry=2
      num_consts=0
      reverse=False
      unroll=1
    ] 0 a
  in (b,) }
```
In the above example, we convert our Python for-loop into `jax.lax.fori_loop`.
This takes arguments for the (integer) start and end of the for loop range, as
well as the function to execute in the body and the starting input value. The
return value of `body_fun` must be the same type as `init_val` and the same
across all iterations. In addition, the input the `body_fun` also takes the
current loop index.

Taking a look at the `jaxpr`, we can see the massive unrolling of operations
has been replaced with a much more compact version, using the `scan` primitive.
This essentially executes the `body_fun` and fixed number of times, carrying
state from one iteration to the next. `scan` compiles `body_fun` and hence
needs a fixed input shape.

> If the number of loops was not static, then we would see a while loop
> primitive instead! There is no primitive of for loop, it is just implemented
> in terms of `scan` or a while loop. 

Let's compiled our less stupid function `less_stupid_fn` and see if we get the
same code out. Even with our fancy primitive functions, XLA should optimise the
function in the same way.
```python
HloModule jit_less_stupid_fn, entry_computation_layout={(f32[2]{0})->f32[2]{0}}, allow_spmd_sharding_propagation_to_output={true}

ENTRY %main.2 (Arg_0.1: f32[2]) -> f32[2] {
  %Arg_0.1 = f32[2]{0} parameter(0), sharding={replicated}
  ROOT %copy = f32[2]{0} copy(f32[2]{0} %Arg_0.1)
}
```
And indeed, we get a single copy operation again.

A similar function exists for while loops named `jax.lax.while_loop`. An equivalent to `less_stupid_fn` would be:
```python
def less_stupid_fn(x):
    y = jnp.copy(x)
    x = jax.lax.while_loop(
        cond_fun=lambda ix: ix[0] < 1000, 
        body_fun=lambda ix: (ix[0]+1, ix[1]*ix[1]),
        init_val=(0, x)
    )
    return y

jax.make_jaxpr(less_stupid_fn)(jnp.array([1.1, -1.1]))
===
Out:
{ lambda ; a:f32[2]. let
    b:f32[2] = copy a
    _:i32[] _:f32[2] = while[
      body_jaxpr={ lambda ; c:i32[] d:f32[2]. let
          e:i32[] = add c 1
          f:f32[2] = mul d d
        in (e, f) }
      body_nconsts=0
      cond_jaxpr={ lambda ; g:i32[] h:f32[2]. let i:bool[] = lt g 1000 in (i,) }
      cond_nconsts=0
    ] 0 a
  in (b,) }
```
Where `body_fun` will continue to be executed so long as `cond_fun` returns
`True`, carrying state between iterations and starting with state `init_val`.

These loops aren't as pretty as Python-level equivalents, but they get the job
done. Remember that it isn't possible to do cross-iteration optimisation with
these loop primitives, as the `body_fun` gets compiled as its own unit. The
same rules apply, try and make `body_fun` as large as possible to give most
context to XLA. If the number of loop iterations is **small and constant** it
may be worth using Python loops instead. For example, you may use a `fori_loop`
to wrap your whole diffusion model during inference, but a regular loop
training an unrolled model for only two, fixed steps.

## Briefly, PyTrees

## Function Transformations

## Conclusion
