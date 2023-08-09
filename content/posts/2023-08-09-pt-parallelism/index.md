---
title: "A Distributed PyTorch Cookbook"
date: 2023-08-09T15:12:30+01:00
draft: true
---

For the past year at [Graphcore](https://graphcore.ai) my working life has been dominated by _getting things to work on many devices_ – mostly using an in-house framework of ours. This taught me a lot about various parallelism techniques in a relatively short space of time. I wrote about this previously [in another blog post earlier this year]().

Now though, I am moving onto pastures new career wise, so I thought it would be a good time to learn how to employ these same techniques in popular frameworks like PyTorch and JAX. Originally I was planning on writing a blog post like this on JAX, however it seems the JAX API for this is still in a state of flux. Furthermore, their documentation currently is pretty swell, and the framework itself is built with relatively transparent parallelism in mind. I will come back to this topic in the future.

I've never really got the same impression from PyTorch's design and I found the documentation on this topic to be _less swell_. This isn't to say it is bad, in fact, I found it to be quite good after coming back to it recently. But I feel the explanations are usually quite long-winded and accompanied by more code than it needs. Having multiple tutorials in different styles for related concepts also makes it hard to compare different methods of parallelism, and how they can be used together.

Therefore, I thought it would be of benefit to the community to share what I have learnt recently about parallelism in PyTorch. The aim is a concise "cookbook" that demonstrates each of the main parallelism methods available in PyTorch, which can be used as a reference for people seeking easy ways to start scaling up their models. For use in production, I would obviously recommend using highly-tuned systems like Megatron-LM and Pythia, and I don't think experts on the topic will get much out of this blog post. However, for students, hobbyists, and researchers who haven't scaled up models before, I hope this can serve as a useful reference!

## The Case of No Parallelism
To even begin of course, we need somewhere to start from.
`TODO: as briefly as possible, describe the setup we are using`

## The Case of Data Parallelism
`TODO: discuss DistributedDataParallel`

## The Case of Sharded Data Parallelism
`TODO: discuss FullyShardedDataParallel`

## The Case of Naive Model Parallelism
`TODO: discuss the most naive case of model parallelism`

## The Case of Pipeline Model Parallelism
`TODO: discuss pipeline parallelism`

## The Case of Tensor Parallelism
Tensor parallelism (TP) is by far the most interesting form of parallelism to me. However, it is also the one I struggle with the most in native PyTorch.

`TODO: discuss tensor parallelism`

## The Case of 3D Parallelism
`TODO: maybe discuss 3D tensor parallelism if I can get it to work in native PyTorch`

## The Case of... making our lives easier
`TODO: discuss nice frameworks around PyTorch to support parallelism in 🤗Accelerate`

---

## Conclusion
