# Dataset 

CHALLENGE 2 - Heart Sound Classification

The task is to produce a method that can classify real heart audio (also known as “beat classification”) 

## Dataset A: four classes

    Normal
    Murmur
    Extra Heart Sound
    Artifact


## Dataset B: three classes

    Normal
    Murmur
    Extrasystole


# Data Description and Organisation

Please use the following citation if the data is used:

@misc{pascal-chsc-2011,
       author = "Bentley, P. and Nordehn, G. and Coimbra, M. and Mannor, S.",
       title = "The {PASCAL} {C}lassifying {H}eart {S}ounds {C}hallenge 2011 {(CHSC2011)} {R}esults",
       howpublished = "http://www.peterjbentley.com/heartchallenge/index.html"}

The audio files are of varying lengths, between 1 second and 30 seconds (some have been clipped to reduce excessive noise and provide the salient fragment of the sound).

Most information in heart sounds is contained in the low frequency components, with noise in the higher frequencies. It is common to apply a low-pass filter at 195 Hz. Fast Fourier transforms are also likely to provide useful information about volume and frequency over time. More domain-specific knowledge about the difference between the categories of sounds is provided below.

# Normal Category
In the Normal category there are normal, healthy heart sounds. These may contain noise in the final second of the recording as the device is removed from the body. They may contain a variety of background noises (from traffic to radios). They may also contain occasional random noise corresponding to breathing, or brushing the microphone against clothing or skin. A normal heart sound has a clear “lub dub, lub dub” pattern, with the time from “lub” to “dub” shorter than the time from “dub” to the next “lub” (when the heart rate is less than 140 beats per minute). Note the temporal description of “lub” and “dub” locations over time in the following illustration:


…lub……….dub……………. lub……….dub……………. lub……….dub……………. lub……….dub…


In medicine we call the lub sound "S1" and the dub sound "S2". Most normal heart rates at rest will be between about 60 and 100 beats (‘lub dub’s) per minute. However, note that since the data may have been collected from children or adults in calm or excited states, the heart rates in the data may vary from 40 to 140 beats or higher per minute. Dataset B also contains noisy_normal data - normal data which includes a substantial amount of background noise or distortion. You may choose to use this or ignore it, however the test set will include some equally noisy examples.

# Murmur Category
Heart murmurs sound as though there is a “whooshing, roaring, rumbling, or turbulent fluid” noise in one of two temporal locations: (1) between “lub” and “dub”, or (2) between “dub” and “lub”. They can be a symptom of many heart disorders, some serious. There will still be a “lub” and a “dub”. One of the things that confuses non-medically trained people is that murmurs happen between lub and dub or between dub and lub; not on lub and not on dub. Below, you can find an asterisk* at the locations a murmur may be.

…lub..****...dub……………. lub..****..dub ……………. lub..****..dub ……………. lub..****..dub …

or

…lub……….dub…******….lub………. dub…******….lub ………. dub…******….lub ……….dub…

**Dataset B** also contains noisy_murmur data - murmur data which includes a substantial amount of background noise or distortion. You may choose to use this or ignore it, however the test set will include some equally noisy examples

# Extra Heart Sound Category (Dataset A)
Extra heart sounds can be identified because there is an additional sound, e.g. a “lub-lub dub” or a “lub dub-dub”. An extra heart sound may not be a sign of disease.  However, in some situations it is an important sign of disease, which if detected early could help a person.  The extra heart sound is important to be able to detect as it cannot be detected by ultrasound very well. Below, note the temporal description of the extra heart sounds:

…lub.lub……….dub………..………. lub. lub……….dub…………….lub.lub……..…….dub…….

or

…lub………. dub.dub………………….lub.……….dub.dub………………….lub……..…….dub. dub……

# Artifact Category (Dataset A)
In the Artifact category there are a wide range of different sounds, including feedback squeals and echoes, speech, music and noise. There are usually no discernable heart sounds, and thus little or no temporal periodicity at frequencies below 195 Hz. This category is the most different from the others. It is important to be able to distinguish this category from the other three categories, so that someone gathering the data can be instructed to try again.

# Extrasystole Category (Dataset B)
Extrasystole sounds may appear occasionally and can be identified because there is a heart sound that is out of rhythm involving extra or skipped heartbeats, e.g. a “lub-lub dub” or a “lub dub-dub”. (This is not the same as an extra heart sound as the event is not regularly occuring.) An extrasystole may not be a sign of disease. It can happen normally in an adult and can be very common in children. However, in some situations extrasystoles can be caused by heart diseases. If these diseases are detected earlier, then treatment is likely to be more effective. Below, note the temporal description of the extra heart sounds:

…........lub……….dub………..………. lub. ………..……….dub…………….lub.lub……..…….dub…….
or
…lub………. dub......………………….lub.…………………dub.dub………………….lub……..…….dub.……
