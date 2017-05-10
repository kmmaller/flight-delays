Flight Cancellation Predictions
-----------------------

Predict whether or not a flight will be cancelled. Data is from U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics which collects on-time performance of US domestic flights. Link to data from 2015 is [here](https://www.kaggle.com/usdot/flight-delays).

Installation
----------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd flight-delays`.
* Run `mkdir data`.
* Switch into the `data` directory using `cd data`.
* Download the data files from U.S. DOT into the `data` directory.  
    * You can find the data [here](https://www.kaggle.com/usdot/flight-delays).
 
* Extract all of the `.zip` files you downloaded.
* Switch back into the `flight-delays` directory using `cd ..`.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.

Usage
-----------------------
* In `settings.py` you can change the list of features and predictions.  
    * Right now `settings.PREDICTIONS` can be either set to ["CANCELLED"] to predict whether a flight was cancelled for any reason for ["CANCELLATION_REASON"] to predict cancellation of a specific reason.  
    * If the latter is chosen, `settings.reason` can be set to "A" (carrier), "B" (weather), "C" (national airsystem), or  "D" (security).
* Run `mkdir processed` to create a directory for our processed datasets.
* Run `python assemble.py` to combine the `Airports` and `Flights` datasets.
    * This will create `processed.txt` in the `processed` folder.
* Run `python annotate.py`.
    * This will create training data from `processed.txt`.
    * It will add a file called `train.csv` to the `processed` folder as well as `new_settings.txt`.
* Run `python predict.py`.
    * This print the accuracy score in the console.

Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* Generate more features in `annotate.py`.
* Switch algorithms in `predict.py`. Instead of Logistic Regression try, perhaps, a Decision Tree?
* Add in cross-validation.
