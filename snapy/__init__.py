"""
This code or file is part of 'Snapy Chatbot' project
copyright (c) 2025, Aymen Brahim Djelloul, All rights reserved.
use of this source code is governed by MIT License that can be found on the project folder.

@author : Aymen Brahim Djelloul
version : 0.2
date : 25.06.2025
license : MIT License

"""

# IMPORTS
from .cli import main
from .snapy import Snapy
# from ._nlp_engine import *
# from ._search_engine import *

__all__ = ["Snapy", "main"]