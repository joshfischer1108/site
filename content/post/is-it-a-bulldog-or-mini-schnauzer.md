---
title: "Is It a Bulldog or Mini Schnauzer?  A Fast.ai implementation."
date: 2022-09-10T11:25:25-05:00
draft: false
---

I recently started working through the Fast.ai course and I’m here to write about what I built and learned and show how you can do the same. To be upfront, I am not a data scientist or an esteemed compute professional. I taught myself how to code with the guidance of others while bartending.  Fast forward 8ish years and here we are talking about building bulldog and mini schnauzer classifiers. **#LevelUp**

I have to admit that I can’t explain how all things work with the project I’m sharing… **_Yet_**. I took this class once before about two years ago, but couldn’t commit the time because of work, life, kids, kids, kids and did I mention kids? The project I’m explaining is in [week 2](https://course.fast.ai/Lessons/lesson2.html) of the revised course. So yes, you can build and learn how to do the same for **FREE**.  I’m going to give the steps needed to rebuild a bulldog and mini schnauzer classification model yourself.  I’ll leave out many of the smaller details of how and why this actually works.  If you want this level of detail, take the [Fast.ai](https://fast.ai) course.

[Here](https://huggingface.co/spaces/joshfischer1108/is-it-a-bulldog-or-mini-schnauzer) is a preview of a running deployment of the model we will be building.
There are a few tools that you will need to make this work. 

- A Kaggle, Google Collab, or Gradient instance.  I’m using Kaggle for this article 
- A little bit of Python knowledge, and Python3 installed
- A HuggingFace Account (optional) - I won’t cover details in this article, but will point you to a good reference.

 ## Set up
You can get a copy of this project by cloning it from my github repo by executing the following in your terminal: 

```
$ git clone git@github.com:joshfischer1108/is-it-a-bulldog-or-a-mini-schnauzer.git
```
You can also view it in your browser here: https://github.com/joshfischer1108/is-it-a-bulldog-or-a-mini-schnauzer

The project we will use has a flat structure, it has one top level folder and all used files are directly beneath it.

```
├── LICENSE
├── README.md
├── app.py
├── bulldog-or-mini-schnauzer.pkl
├── bulldog.jpeg
├── flagged
├── little-puppy.jpeg
├── mini-schnauzer.jpeg
├── requirements.txt
└── snaggle-tooth.jpeg
```


First, let’s make sure you can run the project locally by executing the following:

```
$ pip install -r requirements.txt 
$ python app.py 
```

After running the second line `python app.py`  you should see output similar to the following in your terminal.

```
Running on local URL:  http://127.0.0.1:7860/
```
Once we see this, you should be able to hit that url in your browser to see the following:

![web-view](/img/web-view.png)

 This web page attempts to tell you whether you have uploaded a bulldog or a mini schnauzer photo.  But.. What if you wanted to train your own image classifier?  

 ## Training the model

I put together a Jupyter notebook in Kaggle for you to experiment with.  It was built in a way that you can execute it top to bottom, following the directions in the comments and hopefully all will work as expected.  Again, it is based on lectures I worked through in [Fast.ai](https://fast.ai).

To view and run my Kaggle notebook for building a bulldog and mini schnauzer classifier click here https://www.kaggle.com/code/joshfischer/is-it-a-bulldog-or-a-mini-schnauzer.  Give it an upvote, if you find it worth reading, please.  And!  Be sure to come back to read how to use the model you created and exported.

If you don't want to work through the notebook, no worries.  I have a couple of things I want to point out from the notebook.
1. **We define terms to search with to train our model on**
```python
search_term_1 = "miniature schnauzer"
search_term_2 = "bulldog"
```
2. **We are using DuckDuckGo for our image search.  That function is defined in the notebook as**
 ```python
from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
 ```
 3. **We loop through each one of our search terms adding different phrases to hopefully get better results.**
 **In this case we are looking for pictures in the shade and in the sun.**
 ```python
searches = search_term_1,search_term_2
path = Path('search_results')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)
 ```

4. **We create a `DataLoader` from the images returned from the DuckDuckGo search**
```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```

5. **We download a Resnet18 convolutional neural network and train it on DataLoaders**
```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

6. **We test that our model makes accurate predictions**
```python
print('sending an image of a ' + search_term_1 + ' to the model for classification')
is_term_1,_,probs = learn.predict(PILImage.create(term_1_file_name))
print(f"This is a: {is_term_1}.")
print(f"Probability it's a {search_term_1}: {probs[0]:.4f}\n")

print('sending an image of a ' + search_term_2 + ' to the model for classification')
is_term_2,_,probs = learn.predict(PILImage.create(term_2_file_name))
print(f"This is a: {is_term_2}.")
print(f"Probability it's a {search_term_2}: {probs[1]:.4f}")
```

7. **Finally, we export our model as a `.pkl` file for use in our project**
```python
learn.export('bulldog-mini-schnauzer.pkl')
```
If you leave all the variable names as they are in the notebook,  the end result of running all the steps is a file named `bulldog-mini-schnauzer.pkl`
It’s good to point out that you have just trained a deep learning model, congrats!  This is the exported version of your model that you will need for your project.

You can swap out the old model in the github repo you cloned earlier for by pasting your newly created model into the project.  You can run your project locally, just like before.  

```
$ python app.py 
```

And now you have a web app running locally that is passing image data to your model (the `.pkl` file) at runtime to make a prediction on what type of image you passed to it.  

If you customized the model at all, changed animals, labels, etc. Or if you want to deploy this project on HuggingFace to show off to all your friends.   You will need to make changes to the repo you cloned.  I’m not going to cover the details of how it works in this post.  But you can review [this](https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial) article for more details on how this repo is built. 