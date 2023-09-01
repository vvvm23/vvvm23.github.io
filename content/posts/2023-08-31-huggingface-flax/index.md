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
whole experience though. So if you are thinking, "this guy is a bit dreary, time
to go", don't turn away just yet.

### Positively Speaking

In fact, I feel the experience was quite positive and taught me a lot – just not
much about Flax ironically enough. The maintainers at Huggingface are also very
helpful and do God's work encouraging and nurturing budding open-source
enthusiasts. It is vitally important to keep this open spirit in the machine
learning community, lest we become much more closed off like some other
scientific fields.

So, let me spin these four reasons into a more positive light:

**Regarding vastly overestimating the time and attention to detail required** –
porting models like this is pretty much the only way in machine learning to
actually get accurate and immediate feedback loops on whether you are doing the
right thing. Most of the time, you program a model, train on it, and your loss
may or may not go down. Even if it does go down, is it going down fast enough?
Does it keep going down? What about when I change the problem or the scale of
the model? Maybe your code is right, but these are just bad hyperparameters? Maybe the method itself is flawed and this is a research dead end? The feedback is intrisinctly very fuzzy and sometimes hard to learn from as it isn't always clear what exactly is going wrong.

Meanwhile, with porting models, you have a very clear right or wrong answer. You get a nice immediate feedback loop once you begin writing tests and you get a big fat *bzzzz* when your test fails. It sucks, but it really forces you to get every single thing right. One issue I fixed was that the `jax.numpy.rsqrt` and the `torch.rsqrt` differed by like `0.0001` in certain cases. Over a large enough model, these accumulate into quite a serious numerical difference. A completely boring and annoying issue, but I never would've encountered this had I not tried this project.

**Regarding realising I don't like porting models that much** – this was also a
*positive in the end as it prompted me to think quite deeply about what I
*actually wanted to do in machine learning, or rather, what I was missing. I
*just don't get too excited about porting models, turns out I missed creating
*new models, researching things, and running training experiments. This had a
*knock on effect into me changing job so I can get back to this kind of work.
*This project alone wasn't really the deciding factor, but it did make me think.

I started getting more serious about machine learning after developing a few
model implementations from research papers. Arguably, this is kinda porting
models, but in most cases these models were either entirely unfamiliar to me (in
which I would learn something) or had no existing code. In the latter case, this
provided genuine utility to the community as a reference codebase. Contrast with
straight porting between frameworks, where we don't really get much new
capability except being able to run with a different backend. The actual
"AI-part" is the same.

**Regarding navigating mature and unfamiliar codebases** – I touched on this a
little bit already, but it taught me that when contributing to a large codebase
that has established styles and conventions, I am suspectible to overthinking my
own contributions. However, this wastes a lot of time, so going forward I'll try
to push onwards and _somewhat_ ignore the rest of the library. Easier to ask for
forgiveness than for permission, and not a whole lot can go wrong when you are
working on a branch with good testing. So just trust your gut.

**Regarding turbulent times** – okay, there isn't really a positive here. Moving
house really sucks.

### The Main Point

So, I didn't intend originally for this to be a journey of self-reflection. The
actual main point was to offer some guidance on how to port models between
frameworks in Huggingface transformers. There are a lot of existing docs about
adding totally new models – even dedicated helper tools in the repo – but not
much about converting between frameworks. So without further ado, here is Alex's
little guide to porting models between frameworks.

This isn't really the order in which I did things, and certain pieces of advice
are born out me not doing them the first time, and wasting a bunch of time. This
is the order I would go about doing it if I were to do it again, knowing what I
know.

1. Begin first by obviously picking a model to port. These will be models that have a `modeling_<name>.py` file but not a `modeling_flax_<name>.py`. We need to fully understand what the model architecture actually does, so map out the components of the model and details about them:
    - How are the model input embeddings handled? Not just the embeddings for tokens, but also for positions, token types, and such. Is there any post-processing applied to the embeddings, such as normalisation?
    - How about the output layer? How are the final logits calculated? Are the layers tied to the input embeddings?
    - Looking at the model backbone, what's the structure of each layer? What kind of feedforward layers are they using? What type of attention layer? Any special features of the attention layer? What are the inputs to the attention layer? Where are the normalisation layers and how are the residuals connected? Are the layers arranged purely sequentially or are some in parallel?
    - Basically, obtain a good understanding of all the key components and edge cases in the model. 
2. Given the information about the model, check other Flax implementations in Huggingface transformers. Are there any others that have similar or the same components? For example, I based my Llama implementation heavily off GPT-Neo, which meant a lot of the boilerplate and structure could be copied directly.
    - If there isn't a model that fits perfectly, we can at least reuse components. In any case, whenever you copy model components without changing them (aside from the name) ensure you prepend the line `# Copied from ...` to the code. This keeps these components in sync with one another.
3. We are now ready to begin implementing the new model. Begin from the lowest level components (`FeedForward`, `Embedding`, `Attention`, and any other smaller components) at first.
    - Create small tests for each layer in separate files as you go. They won't make their way into the final release, but they are invaluable for development. Test that the same inputs produce the same results between the reference PyTorch layers and the Flax layers.
    - The tolerance for the tests should be quite strict at this level as errors here will accumulate a lot.
4. Once the low-level components are numerically accurate, move onto higher-level components, such as the larger layers themselves, all the way up to the full model.
    - Again, make tests for these components that test a small, but decent set of hyperparameters. Some issues only arise at certain scales or with certain settings on.
5. Huggingface models usually have variants, such as `...ForCausalLM`, `...ForSentenceClassification` and the like. These also need to be implemented as appropriate for your particular model.
    - There doesn't need to be perfect parity with the PyTorch version, but definitely need to be some variants. In particular, for generative transformer models it almost always makes sense to have the `ForCausalLM` variant.