from model import BaseballPyMC
from data import get_prepared_data
import click
import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.option("--start_year", default=2015, help="First year of data to use")
@click.option("--file_path", default="fit_model.joblib", help="File path to save model")
@click.option("--epochs", default=100, help="Number of epochs to train")
def main(start_year, file_path, epochs):
    player_data = get_prepared_data(start_year)
    # Cleaning NaNs can be better
    train_data = (
        player_data[player_data.Season < 2023].copy().dropna(subset=["next_WAR"])
    )
    model = BaseballPyMC()
    _, _ = model.build_and_fit(
        train_data, validation_split=0.2, n_epochs=epochs, batch_size=64
    )
    model.save_model(file_path)


if __name__ == "__main__":
    main()
