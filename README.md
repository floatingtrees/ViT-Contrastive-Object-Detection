# Implementation of a Fast-RCNN Style with ViTs and Image encodings

Similar to Fast-RCNN, uses 9 achorboxes by default, and uses vision transformers for feature extractors. Code works out of the box, but the training time is long, since ViTs take a long time to learn features. It also uses a CLIP style method to classify images for better generalization, instead of using the standard softmax. To use this repo, download the MSCOCO 2017 dataset and run the following. 

```
git clone https://github.com/floatingtrees/object-detection.git
cd model
```
Go the the training_loop.py script and go to lines 30 and 31. You should see 
```
coco_annotations_file="../../coco2017/annotations/instances_train2017.json"
coco_images_dir="../../coco2017/train2017"
```
Change these to the correct paths for your coco files. 
Install the necesary dependencies with 
```
pip3 install pillow
pip3 install torch torchvision torchaudio
pip3 install ftfy regex tqdm
pip3 install git+https://github.com/openai/CLIP.git
```
OpenAI's CLIP repository is necesary for encoding the text labels as vectors. If you're having trouble downloading it, you can choose to replace the final classification activation function with a softmax and avoid using CLIP. 


Then, run 
```
python3 training_loop.py
```
