from typing import List, Tuple
from random import randint
import numpy as np


class Colorizer():

	def __init__(self):
		pass

	def get_n_colors(self, n: int) -> List[str]:
		"""
		Returns 'n' (most likely different) colors in hex-code.
		:param n: number of desired colors
		:return: list of 'n' colors in hex-code strings
		"""
		return ['#%06X' % randint(0, 0xFFFFFF) for _ in range(n)]

	def hex_str_to_rgb_int(self, hex_str: str) -> Tuple[int, int, int]:
		"""
		This function takes a color in hexadecimal representation
		and returns its rgb-representation.
		:param hex_str: color in hexadecimal representation as a string
		:return: Tuple[int, int, int] - red, green and blue values of the color
		"""
		if hex_str.startswith("#"):
			hex_str = hex_str[1:]

		r = int(hex_str[0:2], 16)
		g = int(hex_str[2:4], 16)
		b = int(hex_str[4:6], 16)

		return r, g, b

def get_ranom_hex_colors(n: int = 1):
    colors = np.random.randint(0,255,3*n)
    hex_colors = ['#%02X%02X%02X' % (colors[i], colors[i+1], colors[i+2]) for i in range(0,3*n, 3)]
    return hex_colors