# IDL_Image_Classification_Project
Repository for the Image Classification Project for Introduction to Deep Learning.

Group members: Sahin Dursun, Ville Puurunen, Lassi Virtanen

Repo link:


Project.py contains our original CNN model, resnet_test.py contains the ResNet model that we ended up using and projecttest.py is our test file which writes the predictions to a csv file. 
Project.py and resnet_test.py will both begin to train when run, and save their checkpoint to a .pt file. Projecttest.py should analyze the test data and save the results to a .csv file when run.
It will print the first 20 predictions on the command line.
NOTE: Projecttest.py must be run on a machine with cuda gpu available, as the currently coded checkpoint("final.pt") checkpoint expects it.

This repository is the "clean" version, link to the actual repository used during development is below:
https://github.com/VilleAR/gitstauts
Proceed at your own risk