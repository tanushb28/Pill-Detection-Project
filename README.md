# Pill detection project summary
## Minimum Scope
The minimum application we have to deliver by Nov 3rd is an executable on some platform (laptop, rpi, smartphone etc.) that uses a camera to detect to a good amount of accuracy whether someone's put a pill in their mouth. Our approach should definitely be lightweight enough in terms of memory and processing power to run on a laptop, ideally it would also work on a smartphone or a large raspberry pi.

## Approach
There's no fixed way to approach this. I'll describe my idea here but this might fail, so we need to be open to thinking about other methods.

### One way of approaching things
- Put a static camera on a table
- Use already trained models by google for face and hand landmark detection to find where the mouth and fingers are each frame
- Somehow use object-detection to detect the pill, especially to check if it's fallen out of the hand.
- Pill-in-mouth is detected by checking if the pill is in front of the mouth and then the mouth closes and the pill can't be detected anymore, or some similar method.

The largest problem with this is that the hand can hide the pill. I thought about this and realised that it's not a CV deficiency, it's a problem with our camera placement: even a human being could be tricked by someone hiding the pill behind their hand and acting like they've taken it. Unlike our computer though, a person can see the pill going into the hand and reason later that the pill must still be in their hand even if they can't see it, unless they can see it fall out. In other words, just like how we can check the pill going into the person's mouth, we can detect it going into their hand and remember this boolean, and we say that the pill is in mouth if the mouth opens and the hand goes over the mouth then the mouth closes etc, and previously we saw the pill go into the hand. It's still possible to trick this system by hiding the pill, like I said, so we'll have to assume the user isn't purposefully trying to trick us. A paper from 2009 describes a similar method, where they avoid detecting the pill at all and instead detect the pill bottle. They also check if the person drinks water after, which we can try to do too.

### Other approaches in the literature (Scroll to the paper section for links)
To remedy the camera angle, I've seen 2 approaches. One is mentioned (but not implemented) in the 2009 paper: use two cameras, one forward and one on the left. The other is discussed in this 2021 paper about wearables: put the camera on the underside of the wrist as part of a smartwatch. This way we always see the pill, and really see it enter the mouth. This feels like more of a complete solution and we can look into this, especially since we can add some way to do ECG's (like other smartwatches do). I don't think this should be our main approach however: firstly because I don't see how we can improve on this paper in some large way on the CV end, so we'd just end up reimplementing their approach without adding anything new, secondly because even remaking their product might be too difficult, because we'd need to make a small plastic watch holder, and gather and label about 200 20-second video clips of people using this watch and taking pills, then train the model (and that's only the first model). Their model architecture and CV details will probably be informative to us. I haven't understood it fully yet though.

## Extended Scope
Some other aspects that would be good to look into:
1. **Detecting type of pill**: This is already a studied problem and there are some smartphone apps that do this.
2. **Look into rotating the camera (related to pill-type-detection)**: A camera looking forward won't have a good angle to see the pills that are lying flat on some plate, so we could try adding a servo to rotate the camera from a looking-down position to a looking-forward position, tracking the hands in some way. This will help having the hands in frame in general.
3. **Smartphone integration**: Initially I thought we could use the phone as a static camera. My main problem with this is that it's hard for old people to use smartphones, and it might be troublesome to put it in place. In the 2021 wearable paper, they use an rpi pico (which is comparable in computation and memory to an arduino) to control the camera and send video information over wifi to a server. I was wondering if it might be useful for a phone to take the place of the server, because the alternative is forcing the user to buy a fairly powerful computer only for this device, which might make the device rather expensive. Whether this makes sense depends on a lot of factors (how and by whom exactly this product will be used, the costs of the camera and the computer), so I don't think it's something we should look into that much now at this stage.
4. **Support for multiple patients**: if this is used somehow in a hospital or in an old-age/nursing home setting, by multiple patients, then it might be useful to detect the face and use this to look into a database for prescriptions, and then check if the pills they're receiving are correct. Ideally we could attach this to some central pill dispenser, so a person would be able to simply go to the camera, get recognized, have their pills fall onto a plate infront of them, have this checked to see if they're the right pills, then take the pills and have this confirmed by the camera. I'm not sure if this would actually be useful in nursing homes because I'm not that knowledgeable about them, and I'm not sure how easy setting and filling up a pill dispenser would be.
    1. Face recognition
    2. Prescription checking
    3. Pill dispenser
    4. Offloading computation onto a central server, similar to the wearables paper. If we look into this we can also add a small model on our weak computer to check if someone's infront it to start sending video information. If we tune it to have much more false positives than false negatives, then this should be good.
5. **Reading text**: I haven't thought about this much but some papers mention reading the text from pill bottles and (printed?) prescriptions and using that to get the types of pills and when they should be taken etc.
6. **Trying to make our approach lightweight enough** to run completely on an arduino (we wouldn't be able to use large off-the-shelf models because they simply wouldn't fit in memory, which for our specific model is 1 MB and in general is under 10 MB, and this isn't even considering processor speed).
7. **Integration with the ECG and the edible pill part of this project**: There's another group working with analyzing ECG's and Neena ma'am and some other people are working on making a pill that we can detect using communication signals, so depending on how things go we can think about joining all of this after we finish with the CV side.

## Finding relevant papers
I had some trouble searching for papers on google initially, and only found papers relating to detecting the type of pill. I realised after that I should specify the keyword "activity recognition". The searches I've used so far are:

    - activity recognition pill
    
    - activity recognition medication
    
    - activity recognition medication adherence
    
Mentioning "computer vision" would probably help as well. Once you've found a relevant paper, you should naturally check the citations and cited-by. They usually have a section at the start talking about previous/related work. There are also separate papers that only review the literature and talk about previous studies. 

### A list of some of them
- 2021 wearable paper:  https://www.mdpi.com/1424-8220/21/11/3594
- 2009 static camera paper: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=662bce94628f97a1cd82f3b12f02a8eb92755e73

Like I said, I found a few papers about type detection but didn't read them very deeply and I didn't bookmark them. So look into any of them. This one might be useful:

- "CNN-Based Pill Image Recognition for Retrieval Systems": https://www.mdpi.com/2076-3417/13/8/5050

Also maybe useful? Part of it talks about pill type detection:

- "Medication adherence management for in-home geriatric care with a companion robot and a wearable device": https://www.sciencedirect.com/science/article/pii/S2352648323000624

## Technical details
### Computer Vision
Disclaimer: I'm not very knowledgeable about CV. 

Kartik looked into adding a 'pill' class for tensorflow object detection (a github project he found: "https://github.com/mepotts/Pill-Detection", an article the github links to: "https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9"). This can detect the type of pill as well. Check in with him for more details. I don't know if this can detect pills that are falling or if that'll be too blurry. I think I also saw some articles about using YOLOv8 to detect pill types. Since pill colors are often quite different from skin color, we can try looking into a non-ML methods in tracking.

The wearable article had a different approach using an LSTM above an object detection model, compared to my more hard-coded approach.

The ready-made face and hand landmark detection algorithms are part of mediapipe. Google has some demo's on their website:
- All demos: https://ai.google.dev/edge/mediapipe/solutions/examples
- Hand landmark detection: https://mediapipe-studio.webapps.google.com/studio/demo/hand_landmarker
- Face landmark detection: https://mediapipe-studio.webapps.google.com/studio/demo/face_landmarker
- Object detection (this includes a guide on how to add custom classes): https://mediapipe-studio.webapps.google.com/studio/demo/object_detector
- Gesture recognition might be useful as well? 

Documentation for all of these: https://ai.google.dev/edge/mediapipe/solutions/guide

The tensorflow blog might also be useful.

### Hardware
We have an Arduino TinyML kit that comes with an arduino nano 33 BLE sense lite and a camera. We can ask for other embedded parts from Neena ma'am.

## Deadlines
I'll need to check the slides for that so I'll update this later. In brief, we'll need to make some planning/scope documents by 31st August and 7th Sept, and there's a presentation (I think) sometime in Sept or October and we have a final submission on November 3rd.

## Github coordination
I think everyone should fork this repo and have a private version, maybe adding a separate directory for a new feature. Then I can merge your private changes into the main repo if you send me a pull request. You can have a new readme.md for each directory, so use that to explain things. You should probably also watch this repo so you get notifs when I change it.
