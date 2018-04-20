"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/lipiodol`
"""

class Config:
	def __init__(self):
		self.dims = [128,128,64]
		self.small_dims = [32,32,16]
		self.nb_channels = 3
		self.aug_factor = 100

		self.png_dir = r"C:\Users\Clinton\Box\FOR CLINTON BOX FOLDER\Lesion Gallery"
		self.base_dir = "C:\\Users\\Clinton\\Documents\\Lipiodol"
		self.ranking_dir = "Z:\\Sophie\\Rankings"
		self.model_dir = "D:\\Lipiodol\\models"
		self.data_xls_path = "D:\\Lipiodol\\Results\\results.xlsx"
		self.fig_dir = r"C:\Users\Clinton\Box\FOR CLINTON BOX FOLDER\Figure drafts\Figure images"
		self.train_data_dir = "D:\\Lipiodol\\npy_data"