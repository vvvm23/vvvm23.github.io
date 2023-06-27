---
title: "Sentence Mining in the terminal"
date: 2023-06-26T13:00:39+01:00
draft: true
---

> An exercise in writing a somewhat short blog

Online, I tend to market myself as an AI / computery guy, mostly creating and following content related to these areas. However, I have a bunch of other interests and passions I like to tinker with – probably too many to actually dedicate lots of time to all of them. However, one of them I have been consistent with for a long while now is language learning, specifically learning Chinese.

It has been quite a long road, starting around late 2018 but with some large gaps and misguided paths along the way. However, in the past couple years I've been very consistent with learning new vocabulary and reading practice. I've basically cracked the formula for learning, at least for me personally, in this regard. I am by no means an expert, but on my recent holiday in Taiwan I managed pretty fine in the reading department – better than I expected. Progressing here now is just a matter of time rather than trying to hack my brain.

However, with listening I was a lot worse than I expected. There were a handful of times I could truly understand what had been said. Most of the time I could get the jist (maybe smiling or laughing at opportune times) but not quick enough to reply in a timely way. I didn't expect to be very good but I was still surprised.

This, of course, is simply a function of time and effort put into listening practice. I think I have spent over a hundred times more time reading than listening, so of course I am worse. Time spent on a skill over a long period of time mostly depends on how consistent you are at practising said skill. How consistent I am mostly comes down to how frictionless it is to get started practising the skill.

For reading, I can just use apps on my phone such as Anki and get little bits of practice in spare minutes here or there. It is extremely easy to do this everyday. Moreover, I have built a sufficient vocabulary base where I can read more interesting content for fun, which also helps stay consistent. Bar literally a handful of days, I have done cuecards every day for the past two years. Listening, however, is typically more labor intensive to get started, but if I want to get better, I need to find a method that makes it easy for lazy-ol' me to be consistent with it.

## Ways of practising listening

Since beginning this journey nearly five years ago, I've explored and discarded a bunch of methods for both reading and listening. I feel a bit part of language learning is just finding a method that works personally for you. Maybe this is a bit of a cope, but when you see on Youtube "I learnt X fluently in Y weeks", the same method may not work for you, but equally it could. Moreover, your goals may be different: levels of fluency; whether you care about writing and reading, or just speaking and listening; types of content, and so on. It is highly personal and takes a lot of trial and error.

I'm happy to say for reading and vocabulary acquisition, I've found my method. However, for listening I am still exploring. Some methods that I have and have not tried include:
- **Textbooks with audio**. These are good when getting started to be honest, but also are not realistic audio examples and the content is often not very engaging.
- **Podcasts**. Probably also good but the interesting stuff is too high level and the lower level stuff is less interesting, which equates to being harder to remain consistent at.
- **Immersion**. This again requires a higher level to really do consistently. But would have amazing gains once I am there.
- **Shadowing**. Just listening to sentences and repeating them back to train both ear and tongue. Very repetitive and boring.
- **Courses**. Stuff like Pimsleur. Probably effective, but expensive and can't tailor the content to stuff I care about, making it again boring.
- **Get a tutor**. This is great too and relatively inexpensive. However, I wouldn't be able to do this every single day. Probably could do in tandem with other methods though.
- **Sentence Mining**. This is basically watching content with native subtitles and finding "$n+i$" sentences (sentences where you understand all but $i$ words – typically $i=1$), creating cuecards from them with audio, and reviewing them using a spaced repetition system. 

Sentence mining is currently what I am using. As it is with subtitles, you can use any reading capabilities to support the listening. Furthermore, you can tailor it totally to your interests by just picking content you like. For example, before I went to Taiwan I watched a lot of videos by Taiwanese Youtubers. Now, as I am about to move in with my girlfriend, I am watching cooking channels so I can learn the terminology so we can cook together in her native language. Out of all the methods I have tried this is the one I have enjoyed the most and I think _could_ become my end-game method.

I first decided to give sentence mining a shot about a year ago after seeing amazing progress using sentence mining for learning Japanese by one of my favourite Youtube channels [Livakivi](https://www.youtube.com/c/livakivi). He has been documenting his process of learning Japanese **every day** for over four years on his Youtube channel, creating 20 new cuecards every day using sentence mining, recently reaching 20,000 cards total. The results are actually amazing, especially his fluency when speaking despite only practising speaking for a total of a few hours over four years.

> This is a reason I tend not to practice speaking, except small phrases just for fun. It seems that speaking skills can follow directly from strong listening abilities.

Despite sentence mining being a promising method, it is also very labor-intensive to do. For each sentence you want to mine, you need to:
- Write the full sentence in Anki.
- Highlight the target words.
- Write the readings (how to pronounce) of the target words. 
- Write the definitions of the target words.
- Record the sentence audio (which can be tricky)
- Optionally, take a screenshot of the content.

> Storing the readings is important in Chinese as the characters don't always give hints on how they are pronounced. It is also a tonal language, so I need to pay attention to tones in the words.

This is quite a lot of steps. Without using tools you can easily spend more time creating cards than actually watching and concentrating on the content and the language. Livakivi, in his videos, uses [Migaku](https://www.migaku.io/) to automate some of the process. Before that, he used an array of tools to somewhat makes things easier, but states that without Migaku he would have burnt out long before reaching 20,000 cards.

> See [his video](https://youtu.be/QBcQJESGQvc) for more details on his process for sentence mining Japanese content.

I don't have god-like levels of discipline to create 20 cards a day manually like I started to do, and soon enough I basically stopped altogether. However, like I mentioned, after coming back from Taiwan I felt somewhat disappointed by my listening abilities and decided to re-approach the problem.

Like I said, in order to put in the time I need to be consistent, and in order to be consistent it needs to be as frictionless as possible. It can never be as easy as reviewing vocabulary cards, but I can try to make it as smooth as possible. And what better way than using ✨programming✨.

## The Things I Did

My goal was to write some kind of script to make sentence mining from videos as easy as possible. For now, I mainly focused on Youtube videos, but the same principles apply to local videos acquired through 🏴‍☠️legitimate means🏴‍☠️. The requirements were as follows:
- **Converts to Anki** in some way, so I can review on all devices, even mobile. Also helps keep consistency by hooking into my existing Anki addiction.
- **Reduce labor cost** to creating cuecards as much as possible, so I can focus on the content I am mining itself.
- Generation script should be **portable**. I travel a lot, so I need to be able to do this on a Mac with a weak CPU as well as on powerful desktops. Mobile would be cool, but not for now.
- **Robust**, nothing kills the mood more than tracebacks.

Contradicting the last point a bit, I decided to practice quickly creating a hacky solution, then gradually iterating on it. I wasn't sure what ideas would work, so I didn't want to prematurely over-engineer a solution only to hate using it.

### Iteration 1 – Basic MVP

The first iteration was quite basic, beginning with creating a script that takes a CSV file with the following fields:
- A sentence with Markdown bold tags highlighting target words.
- Definitions for each of the target words
- Floating point values for the start and end times of the sentence in the content.

This script would then generate readings for each of the targets words using the Python library `pinyin` in each sentence. Then, it would use `youtube-dl` to download the video and use `ffmpeg` to extract audio and screenshots from the target regions. These are then formatted into another CSV file that is importable into Anki.

This is at least easier than using screen capture tools to manually create audio recordings and screenshots, but I am still bottlenecked by copying or writing the sentences, and extracting precise, sub-second timestamps.

### Iteration 2 – Enter Whisper

The next iteration sought to iron out creating the sentences and the time stamps. For this, I stayed on brand a used ✨AI✨ to transcribe videos for me. Again, using `youtube-dl` to download the video audio, I passed the audio to [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax) and parsed the output to generate a CSV file in the same format as the one I created by hand in Iteration 1 – just with an empty definitions field. 

Now, all I need to do is highlight the target words and write their definition; Whisper handles the transcription and timestamps for me. If a particular sentence is too easy or too hard, I can just delete the sentence line. Highlighting and deleting like this is a breeze using Vim.

There are still some issues with this approach however. For one, I found the timestamps were not granular enough, usually just to the second. Secondly, sometimes these timesteps were just straight up inaccurate or had glitchy repeated sentences. Finally, although on GPU this was blazing fast, it was a bit slow on my laptop CPU.

### Iteration 3 – Whisper but slower

The current iteration instead uses [whisper.cpp](https://github.com/sanchit-gandhi/whisper-jax) – a zero dependency, optimised for CPUs Whisper implementation, with limited CUDA support to boot. This makes it much more useable on laptop at the cost of slower desktop performance. In practice this doesn't matter as I can simply do something else as the script runs. Furthermore, I've found the timestamps and transcriptions to be more accurate so far.

The cherry on top is installing a command line version of Google Translate `trans`. It is good for quickly checking the meaning of words by just switching terminal, rather than using my phone dictionary. Google Translate is still a bit unreliable for long sentences, but not bad for single words. If anyone knows of a decent command line English-Chinese dictionary please let me know.

- Limitations and next steps
    - Doesn't integrate with Netflix, where there is a lot of content. Use 🏴‍☠️to resolve.
    - For some reason certain videos just don't work well with whisper. This random cooking channel is just whisper-proof.
    - I can't really stream it as is (think, hour long video) but it isn't really an issue. Just start it running and begin watching when the first CSV is generated.
    - Not always accurate, especially with certain accents. Not really a big deal as I am advanced enough to read the sentence and make corrections, but it does slow things down as I can't 100% trust whisper.
        - However, whisper imo is one of the more robust AI systems out there. I cannot trust ChatGPT to give me accurate answers, but Whisper is correct 99% of the time.
    - God, mobile app would be amazing. But I know zero about front end and mobile dev.

    - I don't automate definitions as often there is a lot of nuance and context to pick the correct definition.
    - Currently the code in this setup is super super messy, it is super fragile and really needs a refactor. Next steps is to smooth it all out.

- Conclusion
    - I probably went totally overkill with this, but it is a fact I am lazy so I need to proactively reduce friction in order to do things. Somewhat contradictory, I know.
    - Still, it is far smoother than what I originally did, and I hope it can help improve my consistency. Time will tell, and this blog post does kinda serve as a dedication that I will commit to doing this more often.
    - In general with computers, I feel it is worth the time to minimise work required wherever possible. That is why I am so into using Linux, but that is for another post. This is just one example of this. Perhaps I fail to remain consistent, but I sure as hell couldn't doing all this work manually. Maybe others can, but I know I cannot.
    - Similarily, I could talk at length about language learning. I have a draft on this but it got crazily long and I lost motivation. Some day though.