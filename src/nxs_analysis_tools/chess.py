'''
Tools for reading scattering datasets collected at CHESS (ID4B).
'''

from nxs_analysis_tools import load_data

class TempDependence():
	'''
	'''
	def __init__(self):
		'''
		'''
		self.tdir = None
		self.temps = None
		self.tdatas = {}

	def set_tdir(self, tdir):
		self.tdir = tdir


	def get_tdir(self):
		return self.tdir

	def load_temps(self):
		'''
		'''
		
		# Identify folders
		tfolders=[] # Empty list to store folder names
		for item in os.listdir(self.tdir):
		  try:
		    tfolders.append(int(item)) # If folder can be converted to int, add it
		  except:
		    pass # Otherwise don't add it
		tfolders.sort() # Sort from low to high T
		tfolders = [str(i) for i in tfolders] # Convert to strings

		# Load .nxs files
		self.tdatas={}
		for tfolder in tfolders:
			for file in os.listdir(os.path.join(self.tdir,tfolder)):
				if file.endswith("hkli.nxs"):
					filepath = os.path.join(self.tdir,tfolder, file)
					print('Loading ' + tfolder + ' K indexed .nxs files...')
					print('Found ' + filepath)
					self.tdata.update({str(tfolder): load_data(filepath)})		            
