from fire import Fire
from arguments import *
from Siamese_trainer import Trainer


def print_status(string):
    line = '*' * 40
    print(line + " " + string + " " + line)


# only train and validation
def train(config=None, trainer=None):
    if config is None or trainer is None:
        print(config, trainer)
        config = ge_config()
        trainer = Trainer(config)

    # Make directory for save logs and model
    prepare_dirs(config)

    # Check resume data
    if config.resume:
        try:
            print(f"load saved config data of model number {config.num_model}")
            load_config(config)
        except ValueError:
            print("[!] config data already exist. Either change the model number, or delete the json file and rerun.")
            return
    else:
        save_config(config)

    # train model
    print_status("Train Start")
    trainer.train()


# only test
def test(config=None, trainer=None):
    if config is None or trainer is None:
        config = ge_config()
        trainer = Trainer(config)

    # test model
    print_status("Test Start")
    trainer.test()


# running all process. download data, data set, data loader, train, validation, test
def run():

    # Make options
    config = ge_config()

    # Make Trainer
    trainer = Trainer(config)

    # train
    train(config, trainer)

    # test
    test(config, trainer)


if __name__ == '__main__':
    Fire(run)
