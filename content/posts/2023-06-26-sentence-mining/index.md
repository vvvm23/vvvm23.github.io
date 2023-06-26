---
title: "Sentence Mining in the terminal"
date: 2023-06-26T13:00:39+01:00
draft: true
---

> An exercise in writing a somewhat short blog

- Introduction
    - Try to market myself as an AI / computery guy.
    - I have other interests though, probably more than is feasible to really delve deep into all of them with
    - One of them is language learning, specifically Chinese.
    - I've got learning reading and vocabulary down, following a standard Anki route.
    - However, I am not too great at listening and speaking. 
        - imo. speaking mostly follows from listening.
    - Mostly down to consistency, it is a lot easier to practice reading everyday than practising listening every day.

- Ways of practising listening
    - I've tried lots and lots of variations and tried to make them work, but I can't get as consistent as Anki grinding and reading.
    - Some ways: shadowing, just talking to random people, textbooks, pimsleur, etc.
    - A popular strategy is **sentence mining** which is essentially watching native content, finding n+1 sentences, recording them, then studying off them.
        - Livakivi is a prime case
    - Most effective thing I have tried thus far, and quite fun as I can target the content exactly as I want it. 
        - For example, before I visited Taiwan I watched a lot of taiwanese youtubers. Now, I am about to move in with my girlfriend, so I am watching cooking channels so I can learn terminology there and we can cook together in her language every day.
    - Problem is, it is quite labour intensive which introduces friction to doing consistently (every day). There are lots of tools to help with this, but I am cheap and could never find a perfect fit.
    - I wanted to make this as easy as possible in order to make me as consistent as possible at it. Anki is easy, I do vocab and reading practice for >99% of days (no exaggeration). sentence mining is far less, once or twice a week at best.

- What I did
    - I had a few requirements:
        - Converts to Anki in some way (review on laptop, desktop, mobile. can review anywhere and anytime. tracks consistency) 
        - reduces work cost of actually creating the cuecards, so I can focus on content and decrease friction (thus, increase consistency) 
        - portable. I travel a lot, so need to at least be able to do it on laptop.

    - There are these components to my cuecards:
        - Sentence with target words in bold
        - Definitions of each of the target words
        - Readings of each of the target words (define what readings mean)
        - Audio from the entire sentence
        - Optional screenshot just for aesthetic purposes.

    - First steps:
        - I created a script that takes a csv file with the following format:
            - sentence with markdown bold tags
            - definitions
            - start and end times of the audio
        - This would then generate readings using the python `pinyin` library, use youtube-dl to download the video (or, local video from totally legal sources), and then use ffmpeg to extract audio and a screenshot automatically. Finally, import into Anki!
        - Easier than using screen / audio capture tools manually and copy pasting into Anki, as well as extracting the readings (harder as chinese is tonal).
        - Still bottlenecked by copying in sentences and getting the precise timestamps (tenth of a second)

    - Second steps:
        - Enter whisper, using the JAX version.
        - This first stage _generates_ the csv file I created manually before.
        - i.e. using whisper, it creates a transcript of the video and the timestamps for each sentence.
        - now, all I need to do is highlight words and write definitions, or delete sentences that are too easy / too hard, as I watch the video.
        - Unfortunately, this JAX version seems to be less accurate and doesn't have granular timesteps. Sometimes I get incorrect timestamps or repeated sentences. It also isn't too fast on CPU, which makes it not portable on my laptop with a crappy CPU.
        - Highlighting and deleting is quite fast with vim.

    - Now?
        - Still whisper, but `whisper.cpp`.
        - Highly portable, zero dependencies, works on CPU well. Limited CUDA support too.
        - Granular timestamps, and typically more accurate predictions.
        - Slower on my desktop with a GPU, but really not that bad.
        - Installed `trans` in command line to quickly look up words.
            - Anyone know a decent command line dictionary? Google translate can be unreliable.

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