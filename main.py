import json
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import warnings
from model import BaseballPyMC

import data

warnings.filterwarnings("ignore")

app = Flask(__name__)

origins = [
    "https://pro.openbb.co",
    "https://excel.openbb.co",
    "http://localhost:1420",
    "https://pro.openbb.dev",
]

CORS(app, origins=origins, supports_credentials=True)

ROOT_PATH = Path(__file__).parent.resolve()


player_data = data.get_prepared_data()

war_model = BaseballPyMC()
war_model.load_model("fit_model.joblib")


@app.route("/")
def read_root():
    return {"Info": "Full example for OpenBB Custom Backend"}


@app.route("/widgets.json")
def get_widgets():
    widgets_path = Path(__file__).parent.resolve() / "widgets.json"
    with widgets_path.open() as f:
        return jsonify(json.load(f))


@app.route("/project-test")
def project_test():
    player = request.args.get("name", "").replace("+", " ")
    if not player:
        return jsonify({"error": "No player name provided"})
    year = request.args.get("year", "2023")
    input_data = player_data[
        (player_data.Name == player) & (player_data.Season == int(year))
    ]
    fig = war_model.plot_prediction_distribution(input_data, player)
    return json.loads(fig.to_json())


@app.route("/player-data")
def hist_player_data():
    player = request.args.get("name", "").replace("+", " ")
    if not player:
        return jsonify({"error": "No player name provided"})
    player = player_data[player_data.Name == player].drop(
        columns=["Name", "next_WAR", "last_WAR"]
    )
    return player.to_json(orient="records")

@app.route("/all-data")
def all_player_data():
    return player_data.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True, port=1234)
