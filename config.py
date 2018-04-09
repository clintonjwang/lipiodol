"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/lipiodol`
"""

class Config:
	def __init__(self):
		self.dims = [128,128,64]
		self.nb_channels = 3
		self.aug_factor = 100

		self.base_dir = "C:\\Users\\Clinton\\Documents\\Lipiodol"
		self.ranking_dir = "Z:\\Sophie\\Rankings"
		self.data_xls_path = "D:\\Lipiodol\\Results\\results.xlsx"

		self.full_img_dir = "D:\\Lipiodol\\npy_data"
