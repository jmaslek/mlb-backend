## Custom MLB Projection and Data Backend

This backend serves as an example backend built in flask that details how to use custom quantitative models in OpenBB.

To replicate this, you start with the cloned repo.  Due to some dependency issues with PyMC that haunt me, I recommend using conda and I have provided a config.

You can run the following to set up your environment:
```bash
conda env create -f environment.yml
conda activate mlb-model
```
To run the backend, you just need to do the following:
```bash
python main.py
```
This will launch a development server on 127.0.0.1:1234.  You can now add this to OpenBB!

I have provided a pre fit model here: `fit_model.joblib`.  When the flask app runs, the model loads from this.

If you wish to rerun anything, I have provided a train script.  For example:
```bash 
python train.py --start_year 2015 --epochs 20
```

By default, this will save to fit_model.joblib.

### Current widgets
I currently provide two widgets.  The first just gets the sample data for a player in a table.

The second is the projection widget.  This widget takes in a player name and a year and returns the projected stats for that player in the next year. 
So using Juan Soto 2024 gives the projections for 2025 using his 2024 stats.

### Future Work
This is as simple as I could make it.  Well slightly more complex than just a simple linear regression model.

Things to do:
- Add more features to the model.  There are like 300 columns in the data, but I pick a few.  Also historical data.
- Data processing.  For simplicity I drop the nan values.  This is not ideal.  We should incorporate more data and maybe use something like a KNN impuation.
- More complex model.  Things like hierarchical models or incorporating information on age or whatever.