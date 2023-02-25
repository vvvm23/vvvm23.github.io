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
[this post for Jax](). I am not a Tensorflow expert, but happy to add links here
if anyone knows of any.

### Pipeline Parallelism

Pipeline Parallelism is used when the entire model is too large to fit on a
single device. The model is split into stages and each stage executed on a
different device, passing activations between neighbouring stages when a
boundary is reached during execution.

For inference, this is extremely good for improving total throughput of the
system as multiple batches can be processed in parallel, with communication only
happening at stage boundaries:

![Inference pipeline](img/pipeline-inference.png)
> Shamelessly taken from [a tutorial from my job.](https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html) Same goes for all images in this section on pipelining.

There is a noticeable **gap** at the start of the above inference pipeline. This is known as the **ramp-up** phase where device utilisation cannot be at 100%. It is clear, that stage 3 cannot begin executing batch 1 until stage 1 and 2 are both done with batch 1. However, in this scheme, once the final stage `N` has received the first batch, we reach the **main phase** and reach maximum utilisation. Once the first stage processes the final batch, we reach the **ramp-down** stage, as their is simply no more data left to execute on.

It should be noted that for best utilisation, the time to execute each stage
should be as balanced as possible. This is because if one stage finishes early,
it may have to wait for the subsequent stage to be ready before passing on its
activations and executing its next batch. This is easy in relatively homogeneous models like transformers, where most layers are typically the same size, but difficult in other models such as Resnets or UNets.

How does this extend to training?

It makes sense for each stage $t$ to also handle its own backwards pass $t$, so
we don't need to copy activations or parameters to another device. However, one
key consideration is that we cannot handle the backwards for a stage $t$,
without having the results for backwards $t+1, \dots, N$ and the forwards for
$t-1, \dots, 1$.

Let's begin with the simplest scheme that meets these conditions:
![Sequential pipeline](img/pipeline-sequential.png)
> Yes, the second set for forward should probably say "B2"

In this scheme, only one batch is in play at a time, leaving all stages but one
idle. At the final stage, we turn around and begin the backwards pass, again
leaving all but one stage idle. Once a full batch is computed (in other words,
after the ramp-down), a weight update is performed. This means the utilisation
is always going to be (at most) $1/N$ of full utilisation. Clearly extremely
inefficient! 

> However, good for debugging purpose!

We can do better with a **grouped pipeline:**
![Grouped pipeline](img/pipeline-grouped.png)

The ramp-up phase almost consists of two "sub-ramp-ups", the first being the same as the inference ramp-up, followed by a ramp-up of the backwards passes which alternate with main-phase forward passes. Once the main-stage is reached, we alternate between forward and backwards passes on all stages.

We can alternate as once a stage $t$ processes the backwards of batch $b$, it can discard activations and accumulate gradients. It is then ready to process the next forward pass for batch $b+1$. Like before, a weight update occurs after a ramp-down phase.

Another approach is to interleave the forwards and backwards passes:
![Interleaved pipeline](img/pipeline-interleaved.png)

So at any given point, half the stages are executing a forward pass, and half the backwards pass. Because of this, the ramp-up and ramp-down phases are much shorter, resulting in a quicker time to maximum utilisation.

The last two schemes have different advantages and disadvantages:
- At any given time, the grouped scheme executes twice as much mini-batches as the interleaved scheme, meaning more memory is required to store activations
- Grouped schemes executes all forward and backwards together, meaning communication is less frequent. Interleaved executes separately, resulting in more communication and also some idle time when forward passes wait for backward passes – which typically take longer than forward passes. Hence, grouped schemes are typically faster than interleaved.
- Interleaved ramp-up and ramp-down time is about twice as fast as grouped, meaning it is quicker to reach full utilisation.

Pipeline parallleism uses much more communication than data parallel, however less than tensor parallelism, which I will discuss in the next section. The amount of communication is not too bad as it is limited to boundaries between stages, meaning regardless of number of replicas, each one will send once, and receive once. The communication can be done in parallel between all replicas.

<!-- TODO: add some links to code-->

### Tensor Parallelism

What about if a single *layer* is too big to fit on a single replica? Say for
example, a particularly large MLP expansion, expensive self-attention layer, or
a large embedding layer. In such cases, the parameters of a layer can be split
across multiple devices. Partial results are then computed on each device before materialising the final result by communicating between the devices.

Take, for example, the MLP block in a standard transformer model that projects a vector $x$ to and from a space of dimension $d$ and $4d$:

$$h = W_2 \cdot f(W_1 \cdot x + b) + b_2 $$
where $f$ is a nonlinear activation function, $W_*$ are the weight matrices, and
$b_*$ are the bias vectors.

To turn this into a tensor parallel layer, do the following:
- Split $W_1$ **row**-wise into `n` pieces, sending one to each of `n` replicas.
- Split $W_2$ **column**-wise into `n` pieces, sending one to each of `n` replicas.
- Split $b_1$ into `n` pieces, as above.

Then, given an input $x$, do on each replica `i`:
- Compute $f(W_1^{(i)} \cdot x + b_1^{(i)})$, resulting in a vector $z_i$ of size $4d/n$.
- Compute $W_2^{(i)} \cdot z_i$, resulting in $\hat{h_i}$ of size $d$

This, naturally, does not give the same result. The next part resolves this:
- Communicate between replicas to compute $\sum^n_{i=1} \hat{h_i} $
- On all replicas add $b_2$, to get the final result $h$ on all replicas.

A nice exercise is to write out mathematically the operations happening here,
and see that we do indeed arrive at the same result. However, I will save myself
~~you~~ the pain here.

<!-- TODO: embedding or attention layer example -->
Another example is an embedding layer in a transformer. We can split the
embedding matrix along the vocabulary dimension, resulting in a shards of shape
$V/n \times d$. All replicas receive the same input tokens and compute the
embedding. However, if any given token falls outside a given replica's valid
range, a zero tensor is instead returned. Then, an all-reduce will rematerialise
the final result, as only one replica (for any given element in the input
sequence) will have a non-zero tensor.

> It should be noted that a distributed all-gather could also be used!

Other examples include splitting attention heads, convolution channels or even
the spatial dimensions themselves other replicas.

Regardless of the context tensor parallelism is applied, it should be noted that
communication between replicas in the same tensor parallel group occur much more
frequently than in data parallel groups, or between pipeline stages. This means
that time spent communicating compared to actually computing a result increases.
This also means that replicas in the same tensor parallel group should be placed
on higher bandwidth interconnect, if possible.  

Depending on the model size tensor parallelism can be totally unavoidable, but
should be avoided if possible! Of course, exceptions are also possible. One case
is where (suppose) you could not use pipeline parallelism, so in order to fit
the whole model onto the available devices, you use tensor parallelism
throughout the model, despite no single layer causing an out of memory. This is
somewhat orthogonal to pipeline parallelism: splitting through the model rather
than across.

<!-- TODO: add some links to code-->

### All together now..

A keen eyed reader may have noticed that these parallelism strategies should
not be incompatible with one another – perhaps they might even use the term
*orthogonal*. Indeed, when we start reaching very large model sizes, or we have
a lot of compute to throw around, we start arranging replicas into hierarchies
and groups.

![Diagram from Microsoft Research's DeepSpeed, showing how different times of parallelism can coexist and complement one another](img/3d-parallelism.png)

Take for example, a system with a data parallel factor of `dp`. Each data
parallel instance will contain an exact copy of the model – simply sending
different batches to each data parallel instance. If we employ pipeline
parallelism with `pp` pipeline stages, and each data parallel instance contains
an exact copy, then each data parallel instance must also contain `pp` pipeline
stages, giving `pp * dp` replicas total.

Adding tensor parallelism to the mix, splitting tensors among `tp` replicas,
each pipeline stage will use `tp` replicas. Naturally, this gives a total
number of replicas of `pp * dp * tp`. This gives a very natural hierarchy of
data parallel at the very top, down to tensor parallelism at the lowest. This
also translates near perfectly to a typical compute cluster.

Recall that data parallelism communications (the gradient all reduce) occur
much less frequently than tensor parallelism communication. Furthermore, larger
clusters typically consist of multiple nodes, where communication between
devices on a single node is much faster than communication between nodes. It
therefore makes sense then to group tensor parallel replicas that need to
communicate together on a single node, and place ones that do not across nodes.
In other words, prioritise placing tensor parallel groups on high-bandwidth
interconnect over data parallel groups.

For example, using IPUs, certain IPUs have more links between them and so give a
higher speed interconnect. Moreover, certain IPUs exist together on the same
motherboard, whereas others only share the same host server, or perhaps even use
different host servers entirely. For GPUs, perhaps certain GPUs are connected
together using NVLink, and others over ethernet between different servers -
perhaps even [over the internet](https://github.com/bigscience-workshop/petals),
swarm style.

### Conclusion

foobar

---

#### Further Reading

foobar

#### References

foobar
