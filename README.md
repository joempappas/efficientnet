# efficientnet
Reimplementation of EfficientNet

#Directions for running experiments

Detailed instructions given in `RunExercises.ipynb`. To run these experiments, 
unzip this directory into a google drive folder. Then you can open the notebook.

You will need to mount the drive to your Google Drive. Then you can read the 
instructions provided in the notebook. 

1. First, run the cells in the Navigate to the directory of this file section. This will 
mount your drive. Please change the file path to the path where you unzipped this directory 
and where this file is located.

2. Second, You can test the various models' accuracy. Run the cell that imports the various modules.
Then you can optionally change the name of the model you want to test. The options are provided in 
the dictionary below the line that you change. Then run the cell. By default, it will just run 1 
test epoch. For my paper results, I ran fo 5 epochs and averaged between them. 

3. Third, you can test the parameters for each model. This is a similar process as above. 

There are git commands at the bottom that can be ignored.


An example of the RayTune Results are also provided. The notebook used to get these 
hyperparameter values is also provided. This is not so much an experiment, so no instructions are
provided, but each cell should be runnable if you choose to check that out.