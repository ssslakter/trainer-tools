from typing import List, Tuple, Union, Optional, Callable, Literal
from functools import partial
import torch, matplotlib
import torchvision as tv
import torch as t, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path