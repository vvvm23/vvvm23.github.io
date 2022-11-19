---
title: "A Brief Overview of Parallelism Strategies in Deep Learning"
date: 2022-11-12T10:11:43Z
draft: false
---

### Introduction

It has been just over two months since I started my first (quote-on-quote) real
job as a fully-fledged graduate. This has taken the form of being an AI
Engineer at Graphcore, a UK-based AI accelerator startup. In quite a short
amount of time, I have learned a great great deal and I am quite grateful for
the opportunity and for their patience – the latter of which is particularly
needed when tutoring the average, fresh compsci graduate.

Chief among what I have learned is a wide array of parallelism strategies. If
you are at least somewhat familiar with Graphcore's **IPU** accelerators, you
will know why. But for the uninitiated, the amount of on-chip memory that can
be directly accessed without hassle on an IPU, is considerably smaller than the
GPUs of today. Though the IPU also has a number of advantages versus a GPU, the
reduced size does mean that parallel execution strategies are a more frequent
occurrence. 

Though that doesn't sound great, if you stop to consider the fast growing model
sizes in the artificial intelligence research community, you will maybe notice
that this *isn't a huge deal* if you are interested in large models. Why is
that? Because now mass parallelism is also required on GPUs. Huge scales are
the great equaliser it seems, and both GPUs and IPUs can do nought next to the
likes of GPT-3, Parti, and friends.

This is exactly the position I found myself in when I landed in Graphcore,
aligning myself very quickly on the large model side of things. Conceptually,
parallelism in deep learning is not tricky, but with implementation I still
have a ways to go. For now though, I will share what I have learned on this
topic. Luckily, the concepts are agnostic in the type of compute device used –
you could even treat everything as desktop CPUs, or a TI-85 graphing calculator
– so this will not be IPU specific nor even framework specific, just high-level
concepts.

### Working with a Single Device

It's helpful to begin with the single device case. One host (a regular PC with
a CPU and memory), one accelerator we want to run our model on, suppose a GPU,
which has its own processing units and memory. During execution, we load
parameters and data onto the device, and create intermediate data on-device
such as activations and gradients. If you are a deep learning practioner you
should be familiar with this scenario.

You will also be familiar with this other scenario:
```
CUDA OUT OF MEMORY
```
For which you probably have a few solutions you immediately reach for (no, not
a bank card). On-device memory is typically used for the following things:
- Storing model parameters.
- Storing activations.
- Storing gradients.
- Storing optimiser states.
- Storing code.

> Having to store gradients and optimiser states on-device is one reason why
> training models uses more memory than just running inference. This is
> particularly true for more advanced (read: memory-hungry) optimisers like
> Adam.

And some solutions typically consist of the following:
- Switch to a lower floating point precision :right_arrow: less bits used per
  tensor.
- Decrease micro-batch size and compensate with gradient accumulation
  :right_arrow: reduces size of activations and gradients.
- Turn on "no gradient" or "inference mode" :right_arrow: only store current
  activations, no gradients stored.
- Sacrifice extra compute for memory savings (ex: recomputation of activations,
  attention serialisation)
- CPU offload :right_arrow: only load tensors onto device when needed,
  otherwise store in host memory.

..among others!

These solutions help a lot in most workflows and can allow for training much
larger models on a single device than you would expect (see [this DeepSpeed
blog]()). However, for one reason or another, this is sometimes just not
practical, or you have money burning a hole in your pocket and fancy renting a
humungous 8xA100 node. 

In such cases, please read on.

### Data Parallelism

Arguably one of this simplest forms of parallelism, data parallelism simply
takes all the devices you have, copies model and optimiser parameters and code
onto all of them, then feeds different data to each device. Gradients are then
synchronised between devices, optimisers step, and parameters are finally
updated. In essence, we use more devices to chew through the dataset faster,
thus obtaining a speedup. It also enables larger (global) batch sizes without
changing the number of gradient accumulation steps (and hence losing speed).

Note though, that the model and optimiser parameters are replicated on all
devices (or replicas). Hence, in order for this parallelism to work, we need to
be able to fit the entire model with a micro-batch size of at least one on a
single replica.

Data parallelism won't directly help fit larger models, but can help pump up
the batch size that you likely had to reduce to squeeze the model in
originally. My first encounter with this was during my master's dissertation,
training a VQ-GAN model, where I could only train on a single device with a
micro-batch size of one, but could obtain a global batch size of 16 with 4
devices and 4 gradient accumulation steps. 

Data parallelism also is pretty communication light, as we only need to
all-reduce the gradients whenever we update the optimiser, which could be quite
infrequent depending on the number of gradient accumulation steps. This also
reduces the need for high-speed interconnect between replicas, as communication
is infrequent.

To use data parallelism in your code see [this post for PyTorch fans]() and
[this post for Jax](). I can't really comment on the best Tensorflow source,
but it surely is implemented.

### Tensor Parallelism

foobar

### Pipeline Parallelism

foobar

### All together now..

foobar
