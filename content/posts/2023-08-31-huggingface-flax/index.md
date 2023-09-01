---
title: "Porting PyTorch Huggingface Transformers to Flax"
date: 2023-08-31T21:08:07+01:00
draft: true
---

After writing about Jax and Flax in some previous blog posts, I wanted to learn
a little bit more about the latter as I hadn't really delved all that deeply
into it. To this end, I decided to pick a Flax-related project that I could
complete in a reasonable length of time. I decided to try porting a _few_
PyTorch models in the all-too-well-known library Huggingface (🤗) Transformers
to Flax – its less loved backend.

<!-- TODO: verify two months -->
About two months later, I have just finished only one model port, Llama, which
to be fair is quite a hefty model. Bit late for the Llama hype, but roughly in
time for Code Llama hype I guess.

There are a number of reasons why this took so long, and most are my own fault
really. Firstly, I simply vastly underestimated the time and attention to detail
required to implement a model that precisely matches a reference implementation.
I think most people reading this could probably code up a transformer model
pretty fast, but to make one that matches _exactly_ an implementation in a
different framework is quite tricky. Originally, I had planned to do _at least
one model_ a week, which I can see now was pretty foolhardely.

I think the second reason is that I didn't realise in advance that I _kinda don't like porting models to other frameworks._ I basically already do this for my day job, so motivating myself to work on this in the evenings was hard at times.

The third is that navigating a mature and unfamiliar codebase is quite daunting.
Huggingface has quite a friendly API that I know quite well – making its way
into pretty much every machine learning related project I do. However,
contributing to such a mature codebase with its own opinions on what is correct, causes a bit of anxiety in this relatively junior machine learning engineer. I basically kept questioning myself at every stage about whether something was right, best, or what the core maintainers would want, rather than simply just coding the damn thing and apologising for mistakes later.

Finally, I've simply had a turbulent time recently with moving house and moving job. I still try to do personal projects in times of unrest but this one in particular was the first to be deprioritised. Take care of yourself first!

This isn't a blog post to rant, complain, or generally be negative about the
whole experience though. In fact, I feel it was quite positive and taught me a
lot – just not much about Flax ironically enough. The maintainers at Huggingface
are also very helpful and do God's work encouraging and nurturing budding
open-source enthusiasts. It is vitally important to keep this open spirit in the
machine learning community, lest we become much more closed off like some other
scientific fields.

So, let me spin these four reasons more positively:

**Regarding vastly overestimating the time and attention to detail required** –
porting models like this is pretty much the only way in machine learning to
actually get accurate and immediate feedback loops on whether you are doing the
right thing. Most of the time, you program a model, train on it, and your loss
may or may not go down. Even if it does go down, is it going down fast enough?
Does it keep going down? What about when I change the problem or the scale of
the model? Maybe your code is right, but these are just bad hyperparameters? Maybe the method itself is flawed and this is a research dead end? The feedback is intrisinctly very fuzzy and sometimes hard to learn from as it isn't always clear what exactly is going wrong.

Meanwhile, with porting models, you have a very clear right or wrong answer. You get a nice immediate feedback loop once you begin writing tests and you get a big fat *bzzzz* when your test fails. It sucks, but it really forces you to get every single thing right. One issue I fixed was that the `jax.numpy.rsqrt` and the `torch.rsqrt` differed by like `0.0001` in certain cases. Over a large enough model, these accumulate into quite a serious numerical difference. A completely boring and annoying issue, but I never would've encountered this had I not tried this project.

**Regarding realising I don't like porting models that much** – this was also a positive in the end as it prompted me to think quite deeply about what I actually wanted to do in machine learning, or rather, what I was missing. I just don't get too excited about porting models, turns out I missed creating new models, researching things, and running training experiments. This had a knock on effect into me changing job so I can get back to this kind of work. This project alone wasn't really the deciding factor, but it did make me think.

**Regarding turbulent times** – okay, there isn't really a positive here