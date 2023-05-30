'''
Tools for reading scattering datasets collected at CHESS (ID4B).
'''

from nxs_analysis_tools import load_data, plot_slice, Scissors
from nexusformat.nexus import NXentry
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

class TempDependence():
	'''
	'''
	def __init__(self):
		'''
		'''
		self.datasets=NXentry()
		self.folder=None
		self.temperatures=None

	def get_folder(self):
		'''
		'''
		return self.folder

	def clear_datasets(self):
		self.datasets=NXentry()

	def load_datasets(self, folder, file_ending='hkli.nxs', temperatures_list=None):
		'''
		'''
		self.folder = os.path.normpath(folder)

		temperature_folders=[] # Empty list to store temperature folder names
		for item in os.listdir(self.folder):
		  try:
		    temperature_folders.append(int(item)) # If folder name can be converted to int, add it
		  except ValueError:
		    pass # Otherwise don't add it
		temperature_folders.sort() # Sort from low to high T
		temperature_folders = [str(i) for i in temperature_folders] # Convert to strings

		self.temperatures = temperature_folders

		if temperatures_list is not None:
			temperature_folders = [str(t) for t in temperatures_list]
		else:
			temperature_folders = self.temperatures

		self.temperatures = temperature_folders

		# Load .nxs files
		for temperature in temperature_folders:
			for file in os.listdir(os.path.join(self.folder,temperature)):
				if file.endswith(file_ending):
					filepath = os.path.join(self.folder,temperature, file)
					print('-----------------------------------------------')
					print('Loading ' + temperature + ' K indexed .nxs files...')
					print('Found ' + filepath)
					self.datasets[temperature] = load_data(filepath)

	def cut_data(self, center, window, axis=None):
		self.scissors = [Scissors() for _ in range(len(self.temperatures))]
		for i,T in enumerate(self.temperatures):
			print("-------------------------------")
			print("Cutting T = " + T + " K data...")
			self.scissors[i].set_data(self.datasets[T])
			self.scissors[i].cut_data(center, window, axis)

		self.linecuts = [scissors.linecut for scissors in self.scissors]
		return self.linecuts

	def plot_linecuts(self, **kwargs):
	    fig, ax = plt.subplots()
	    
	    # Get the Viridis colormap
	    cmap = cm.get_cmap('viridis')
	    
	    for i, linecut in enumerate(self.linecuts):
	        x_data = linecut[linecut.axes[0]].nxdata
	        y_data = linecut[linecut.signal].nxdata
	        ax.plot(x_data, y_data, color=cmap(i / len(self.linecuts)), **kwargs)
	    
	    plt.show()
