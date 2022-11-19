---
title: "A Brief Overview of Parallelism Strategies in Deep Learning"
date: 2022-11-12T10:11:43Z
draft: false
---

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
