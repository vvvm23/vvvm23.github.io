---
title: "A Distributed PyTorch Cookbook"
date: 2023-08-09T15:12:30+01:00
draft: true
math: true
---

During my time at [Graphcore](https://graphcore.ai) my working life was dominated by _getting things to work on many devices_ – mostly using an in-house framework of ours. This taught me a lot about various parallelism techniques in a relatively short space of time. I wrote about this previously [in another blog post earlier this year]().

Now though, I have moved onto pastures new career wise, so I thought it would be a good time to learn how to employ these same techniques in popular frameworks like PyTorch and JAX. Originally I was planning on writing a blog post like this on JAX, however it seems the JAX API for this is still in a state of flux. Furthermore, their documentation currently is pretty swell, and the framework itself is built with relatively transparent parallelism in mind. I will come back to this topic in the future, but other resources online do a pretty good job explaining it. 

I never really got the same impression from PyTorch's design, and I find the documentation on this topic to be _less swell_. This isn't to say it is bad, in fact I found it to be quite good after coming back to it recently. But I feel the explanations are usually quite long-winded and accompanied by more code than needed. Having multiple tutorials in different styles for related concepts also makes it hard to compare different methods of parallelism and how they can be used together.

Therefore, I thought it would be of benefit to the community to share what I have learnt recently about parallelism in PyTorch. The aim is a concise "cookbook" that demonstrates each of the main parallelism methods available in PyTorch, which can be used as a reference for people seeking easy ways to start scaling up their models. For use in production, I would obviously recommend using highly-tuned systems like Megatron-LM and Pythia, and I don't think experts on the topic will get much out of this blog post. However, for students, hobbyists, and researchers who haven't scaled up models before, I hope this can serve as a useful reference!

## The Case of No Parallelism
To even begin, we need somewhere to start.

Our task is to train a small transformer decoder on the
[TinyStories](https://arxiv.org/abs/2305.07759) dataset to simply predict the
next token given the previous ones. Where possible I essentially follow the
process in that paper.

For the sake of brevity, I won't write out all the code but you can find it
[here](https://github.com/distributed-pytorch-cookbook) instead. Where
narratively appropriate, I will include snippets of code in the writing.

You can find model code [here](), data loading and processing [here](), and the
main train script [here](). I've tried to keep each training script separate to
make each form of parallelism easy to compare with others.

The objectives when training with all forms of parallelism is as follows:
- Achieve similar (ideally identical) evaluation loss to the "no parallelism" baseline.
- Obtain improvements in training throughput expected from employing the target mode of parallelism.

Let's take a look at the main train script.

## The Case of Data Parallelism
`TODO: discuss DistributedDataParallel`

- Distributed data parallelism (DDP)
- Data parallelism involves splitting up the input batch across multiple
accelerators, allowing for multiple minibatches to be processed at the same
time.
- This is useful in two cases:
    - One, if we had to reduce the batch size in the no parallelism case to get the model to fit on one device, and we want to increase the batch size again (by splitting the full batch over many devices).
        - This could be equally be achieved with gradient accumulation, but would be slower to train.
        - Data parallelism is kinda like gradient accumulation except we do all the steps in parallel across multiple GPUs.
    - Simply to train faster, by churning through the dataset faster.
- One question to answer is that, if each device gets a different batch, therefore a different loss, and therefore different gradients: won't we just have a different model on each device?
- Yes. Which is why there are two additional steps:
    - At the start of training, copy identical model and optimiser parameters to each accelerator OR initialise them in exactly the same way (with the same PRNG seed). This ensures each process begins from the same set of parameters.
    - During training, when each process has computed all its gradients, perform an all-reduce to average the gradients across all devices.
        - The latter point is why we don't get perfectly linear speedup with number of GPUs.
- As the gradients are now identical across all processes, the optimiser updates are also identical and hence the parameters remain in sync without explictly communicating the parameters.
- How does this look in code? (using torchrun)
    - There is some extra boilerplate associated with DDP. Discuss `distributed.init_process_group` and the like.
    - Moving our model to the correct rank
    - Wrapping our model in `DistributedDataParallel`
    - How to execute with torchrun
        - Mention it is possible to execute without torchrun, but need some extra boilerplate
- Show me the results!
- Caveat, mention `DataParallel` and why it is not preferred.
- Some extra tidbits, like how to only execute stuff on certain processes (printing, logging, checkpointing) and barriers. DistributedSampler?

`TODO: https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer`

## The Case of Sharded Data Parallelism
`TODO: discuss FullyShardedDataParallel`
`TODO: replace with all gather, all scatter terminology`

- Fully sharded data parallelism (FSDP)
- In many ways quite similar to DDP as we are still splitting the training batch across devices.
- Recall that for DDP the gradients, optimiser states, and model parameters were all copied to device. FSDP can instead shard these across all devices, so each process maintains a slice of the full thing.
- Before they can be used, the full tensor will be need be gathered together from across all devices so that each device has the full tensor again. Once we are finished with the tensor, we can return to only storing a shard of it by scattering it across devices.
- Typically, the states, gradients and parameters are gathered and scattered in groups or even not sharded at all. How the groups are defined is usually defined by a policy.
- In code, there are two main ways to apply FSDP. 
    - The first is basically a drop-in replacement for DDP where you only need to wrap the model differently, and specify some arguments to an auto wrap policy.
    - The second involves manually defining where to wrap layers into FSDP groups. This can be done by annotating the model definition or writing your own policy.
    - We will cover the first for now.
    - First, and building on the DDP example, create and define your auto wrap policy. We will base off size_based_auto_wrap_policy for now
        - `TODO: polciies are here https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/wrap.py`
        - the transformer one could be useful.
    - Wrap your model in the FSDP wrapper and pass in your policy.
    - Execute in same way with torchrun.
- Show me the results?
- Extra tidbits again, such as CPU offload, different policies, different sharding strategies (`fsdp.ShardingStrategy`)
    

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
`TODO: discuss nice frameworks around PyTorch to support parallelism in 🤗Accelerate and Lightning`

---

## Conclusion
