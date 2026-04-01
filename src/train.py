"""
A very fake training module for demo purposes
"""
import os

def train():
    """
    An extremely useless training function for demo purposes
    """
    print('Current training mode is {}'.format(os.environ.get('TRAINING_MODE', 'not set')))


if __name__ == '__main__':
    train()
