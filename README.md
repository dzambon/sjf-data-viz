# Machine learning for large datasets visualization

Study Week -- Fascinating Informatics -- [www.sjf.ch](www.sjf.ch)  
September 2020

Samuel Baumgartner (Gymnasium Oberwil)  
Michelle Lebo (Gymnasium NMS Bern)  
Supervised by Daniele Zambon (Università della Svizzera italiana)  

**Abstract**: 
Can we perceive dimensions beyond the third? Can we, as aspiring data scientists, make visual representations of large and multi-dimensional data sets, that seem too complex for humans to have a sense of?
In this project, we implement machine learning algorithms to collections of images, non-numeric entities like words, or data from social networks, and we let the machine find patterns that allow us to draw intuitive visualizations and to navigate through such huge data sets.

## Road map

**Preliminaries**

- Data visualization in Python

**Music representation and visualization**

- Download data from Spotify through the API
- Data cleaning
- 2D and 3D representations
- Visualization of the representations

**Playlist creation**

- Explore the dataset 
- Create a playlist from walks
- Upload the playlist on Spotify through the API


## Software setup

### Python
```
py -m pip install virtualenv
py -m virtualenv venv
source venv/bin/activate
```

### Git repository of the project
```
git clone https://github.com/dzambon/sjf-data-viz
py -m pip install -e .
```

### PyCharm IDE (Optional)
Download and open the project. Select the created virtual environment in the interpreter settings.

### FFMPEG (Optional)
Download `ffmpeg` from [https://ffmpeg.zeranoe.com/builds/](https://ffmpeg.zeranoe.com/builds/) and add the bin folder to `PATH` variable:
```
export PATH="...ffmpeg.folder.../bin:$PATH"
```
