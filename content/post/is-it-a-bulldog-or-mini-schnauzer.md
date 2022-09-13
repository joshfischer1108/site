---
title: "Is It a Bulldog or Mini Schnauzer?  A Fast.ai implementation."
date: 2022-09-10T11:25:25-05:00
images:
- feature-web-view.png
draft: false
---

## TL;DR
* View and run my [Kaggle notebook for building a bulldog and mini schnauzer classifier](https://www.kaggle.com/code/joshfischer/is-it-a-bulldog-or-a-mini-schnauzer)
  * Give it an upvote, if you find it worth reading, please.  And!  Be sure to come back to read how to use the model you created and exported.
* [Here is a preview of a running deployment](https://huggingface.co/spaces/joshfischer1108/is-it-a-bulldog-or-mini-schnauzer) of the model we will be building.
  * _Go ahead, give it a heart thingy_
* [Here is the Githbub Repo](https://github.com/joshfischer1108/is-it-a-bulldog-or-a-mini-schnauzer)
  * _Give this a star while you are there, please_

## The Deets
I recently started working through the Fast.ai course and I’m here to write about what I built and learned. And, show how you can do the same. To be upfront, I am not a data scientist or an esteemed compute professional. I am self taught with the guidance of others.  I came from 10+ years bartending and waiting tables. Fast forward 8ish years and here we are talking about building bulldog and mini schnauzer classifiers. **#LevelUp**

I have to admit that I can’t explain how all things work with the project I’m sharing, **_yet_**. I took this class once before about two years ago, but couldn’t commit the time because of work, life, kids, kids, kids and did I mention kids? The project I’m explaining is in [week 2](https://course.fast.ai/Lessons/lesson2.html) of the revised course. So yes, you can build and learn how to do the same for free.  I’m going to give the steps needed to rebuild a bulldog and mini schnauzer classification model yourself.  I’ll leave out many of the smaller details of how and why this actually works.  If you want this level of detail, take the [Fast.ai course](https://course.fast.ai/).

[Here is a preview of a running deployment](https://huggingface.co/spaces/joshfischer1108/is-it-a-bulldog-or-mini-schnauzer) using the model we will be building.
There are a few tools that you will need to make this work. 

- A smidgen of Jupyter Notebook knowledge.  
  - All I really need you to know is how to execute a cell. This can be done by pressing `shift + enter` when you are in a cell.  [There are more in depth instructions here](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html)
- [A Kaggle account](https://www.kaggle.com/) 
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

![web-view](/site/img/feature-web-view.png)

 This web page attempts to tell you whether you have uploaded a bulldog or a mini schnauzer photo.  But.. What if you wanted to train your own image classifier?  

 ## Training the model

I put together a Jupyter notebook in Kaggle for you to experiment with.  It was built in a way that you can execute it top to bottom, following the directions in the comments and hopefully all will work as expected.  Again, it is based on lectures I worked through in [Fast.ai](https://fast.ai).

To [view and run my Kaggle notebook for building a bulldog and mini schnauzer classifier click here](https://www.kaggle.com/code/joshfischer/is-it-a-bulldog-or-a-mini-schnauzer).  Give it an upvote, if you find it worth reading, please.  And!  Be sure to come back to read how to use the model you created and exported.

If you don't want to work through the notebook, no worries.  I have a couple of things I want to point out from the notebook.
1. **We define the terms that will be used to search for images. These images will be used to train our model**
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
    * **_Side Note_**: A `DataLoader` [represents a Python iterable over a dataset in the Fast.ai API.](https://docs.fast.ai/data.load.html#dataloader)
    * **_Side Note Again_**: The Fast.ai `DataLoader` is a wrapper around [the PyTorch `DataLoader`](https://pytorch.org/docs/stable/data.html#module-torch.utils.data)
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
    * **_Another Side Note_**: Unsure what `Resnet18` is?  In short, it's used for building out many computer vision tasks.
    [You can find more details here](https://www.mathworks.com/help/deeplearning/ref/resnet18.html#:~:text=ResNet%2D18%20is%20a%20convolutional,%2C%20pencil%2C%20and%20many%20animals.) 
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
This python may be a bit hard to read. Below is what the output will look like
```
sending an image of a miniature schnauzer to the model for classification
This is a: miniature schnauzer.
Probability it's a miniature schnauzer: 1.0000

sending an image of a bulldog to the model for classification
This is a: bulldog.
Probability it's a bulldog: 0.9996
```

7. **Finally, we export our model as a `.pkl` file for use in our project**
```python
learn.export('bulldog-mini-schnauzer.pkl')
```

If you leave all the variable names as they are in the notebook,  the end result of running all the steps is a file named `bulldog-mini-schnauzer.pkl`
It’s good to point out that you have just trained a deep learning model, congrats!  This is the exported version of your model that you will need for your project. If you change the name of the `.pkl` file you will need to update your `app.py` file with the appropriate name.  [Here is a direct link to the variable name in Github.](https://github.com/joshfischer1108/is-it-a-bulldog-or-a-mini-schnauzer/blob/1c91b0ee10592e7a7dce4cc323bbe565dfc01bd0/app.py#L6)

 ## Adding the new model back into our project

Remember when we did that `git clone` command earlier?  It looked like
```
$ git clone git@github.com:joshfischer1108/is-it-a-bulldog-or-a-mini-schnauzer.git
```
Ok, good.  What ever folder you cloned that repo that repo in, make sure you get back there in your terminal to `cd` into the root of the project.
An example of how to do so is below:
```
$ cd is-it-a-bulldog-or-a-mini-schnauzer
```

if you list the contents of the directory you just cloned you should see an already existing file named `bulldog-mini-schnauzer.pkl`. Replace that one with the new `bulldog-mini-schnauzer.pkl` you just downloaded from Kaggle. You can run your project locally, just like before.  

```
$ python app.py 
```

And now you have a web app running locally that is passing image data to your model (the `.pkl` file) at runtime to make a prediction on what type of image you passed to it.  

If you customized the model at all, changed animals, labels, etc. Or if you want to deploy this project on HuggingFace to show off to all your friends.   You will need to make changes to the repo you cloned.  I’m not going to cover the details of how it works in this post.  But you can review [this article](https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial) for more details on how this repo is built. 