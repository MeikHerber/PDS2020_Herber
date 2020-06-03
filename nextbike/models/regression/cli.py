import click
from ..general import prepare_dataset
from ..regression import modelling


def train_and_save(model_name, csv_name='dortmund'):
    x_train, x_test, x_train_scale, x_test_scale, y_train, y_test = prepare_dataset(
        f'{csv_name}_preprocessed_regression')
    modelling(model_name, x_train_scale, x_test_scale, y_train, y_test, export=True)


@click.command()
@click.option('--model', default='OLS_Regression', help='Train the model.')
def main(model):
    train_and_save(model_name=model)


if __name__ == '__main__':
    main()
